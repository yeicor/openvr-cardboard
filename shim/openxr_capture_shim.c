/*
 * OpenVR Cardboard OpenXR capture shim.
 *
 * A small LD_PRELOAD library that observes a real OpenXR application and
 * forwards what it renders and what it does to the OpenVR Cardboard desktop
 * app. It is test/demo tooling: it exists so the recorded end-to-end test can
 * show the actual eye textures, the grab action and the haptic feedback of an
 * unmodified OpenXR application (the Monado openxr-simple-example).
 *
 * Captured side-by-side JPEG frames are streamed over a loopback TCP socket to
 * the desktop app. The wire format is deliberately tiny and local-only:
 *
 *   magic "OVXS" | type:u8 | reserved:u8[3] | payload_length:u32le | payload
 *
 * Frame payload: sequence:u32 | timestamp:f64 | width:u32 | height:u32 | JPEG
 * Event payload: UTF-8 JSON.
 *
 * Environment:
 *   OVRC_CAPTURE_HOST      desktop host (default 127.0.0.1)
 *   OVRC_CAPTURE_PORT      desktop TCP port (default 39102)
 *   OVRC_CAPTURE_EYE_WIDTH per-eye capture width (default 640)
 *   OVRC_CAPTURE_EYE_HEIGHT per-eye capture height (default 720)
 *   OVRC_CAPTURE_FPS       maximum captured frames per second (default 30)
 *   OVRC_CAPTURE_QUALITY   JPEG quality 20..95 (default 80)
 *   OVRC_CAPTURE_GAMMA     sRGB encode exponent, 0 disables (default 2.2)
 *   OVRC_CAPTURE_DEBUG     set to 1 to log per-eye capture diagnostics
 */

#define _GNU_SOURCE
#define XR_USE_GRAPHICS_API_OPENGL 1

#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <arpa/inet.h>
#include <endian.h>
#include <dlfcn.h>
#include <math.h>
#include <stdarg.h>
#include <errno.h>
#include <stdio.h>
#include <jpeglib.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define OVRC_EXPORT __attribute__((visibility("default")))

/* Minimal GL types/constants so the shim only needs libGL at runtime. */
typedef unsigned int GLuint;
typedef unsigned int GLenum;
typedef int GLint;
typedef int GLsizei;

#define GL_TEXTURE_2D 0x0DE1
#define GL_RGBA 0x1908
#define GL_UNSIGNED_BYTE 0x1401
#define GL_TEXTURE_BINDING_2D 0x8069
#define GL_PACK_ALIGNMENT 0x0D05
#define GL_PACK_ROW_LENGTH 0x0D02
#define GL_PACK_SKIP_ROWS 0x0D03
#define GL_PACK_SKIP_PIXELS 0x0D04
#define GL_PIXEL_PACK_BUFFER 0x88EB
#define GL_PIXEL_PACK_BUFFER_BINDING 0x88ED

#define MAX_SWAPCHAINS 16
#define MAX_ACTIONS 128
#define MAX_PATHS 512
#define MAX_FRAMES 3
#define MAX_EVENTS 128
#define FRAME_MAGIC "OVXS"
#define FRAME_HEADER_SIZE 12
#define FRAME_PAYLOAD_HEADER 20

struct swapchain_info
{
	XrSwapchain handle;
	uint32_t width;
	uint32_t height;
	uint32_t count;
	uint32_t last_index;
	GLuint *textures;
	int last_valid;
};

struct action_info
{
	XrAction handle;
	char name[XR_MAX_ACTION_NAME_SIZE];
};

struct path_info
{
	XrPath path;
	char str[XR_MAX_PATH_LENGTH];
};

struct frame_job
{
	uint32_t sequence;
	double timestamp;
	uint32_t width;
	uint32_t height;
	uint8_t *rgb;
};

struct event_job
{
	char *json;
	struct event_job *next;
};

typedef void (*PFN_glGetTexImage)(GLenum, GLint, GLenum, GLenum, void *);
typedef void (*PFN_glGetIntegerv)(GLenum, GLint *);
typedef void (*PFN_glPixelStorei)(GLenum, GLint);
typedef void (*PFN_glBindTexture)(GLenum, GLuint);

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cond = PTHREAD_COND_INITIALIZER;

static char g_host[256] = "127.0.0.1";
static int g_port = 39102;
static int g_eye_width = 640;
static int g_eye_height = 720;
static int g_fps = 30;
static int g_quality = 80;
static double g_gamma = 2.2;
static uint8_t g_gamma_lut[256];
static int g_debug;

static struct swapchain_info g_swapchains[MAX_SWAPCHAINS];
static int g_swapchain_count;

static struct action_info g_actions[MAX_ACTIONS];
static int g_action_count;

static struct path_info g_paths[MAX_PATHS];
static int g_path_count;

static struct frame_job g_frames[MAX_FRAMES];
static int g_frame_head;
static int g_frame_count;

static struct event_job *g_event_head;
static struct event_job *g_event_tail;
static int g_event_count;

static int g_socket = -1;
static int g_initialized;
static uint32_t g_sequence;
static double g_started_at;
static double g_last_capture;
static double g_last_haptic[2] = {-1.0, -1.0};
static double g_last_grab[2] = {-1.0, -1.0};
static float g_last_grab_value[2] = {-1.0f, -1.0f};

static PFN_glGetTexImage p_glGetTexImage;
static PFN_glGetIntegerv p_glGetIntegerv;
static PFN_glPixelStorei p_glPixelStorei;
static PFN_glBindTexture p_glBindTexture;

static XrResult (*real_xrCreateSwapchain)(XrSession, const XrSwapchainCreateInfo *, XrSwapchain *);
static XrResult (*real_xrDestroySwapchain)(XrSwapchain);
static XrResult (*real_xrEnumerateSwapchainImages)(XrSwapchain, uint32_t, uint32_t *,
                                                   XrSwapchainImageBaseHeader *);
static XrResult (*real_xrAcquireSwapchainImage)(XrSwapchain, const XrSwapchainImageAcquireInfo *,
                                                uint32_t *);
static XrResult (*real_xrEndFrame)(XrSession, const XrFrameEndInfo *);
static XrResult (*real_xrCreateAction)(XrActionSet, const XrActionCreateInfo *, XrAction *);
static XrResult (*real_xrStringToPath)(XrInstance, const char *, XrPath *);
static XrResult (*real_xrGetActionStateFloat)(XrSession, const XrActionStateGetInfo *,
                                              XrActionStateFloat *);
static XrResult (*real_xrApplyHapticFeedback)(XrSession, const XrHapticActionInfo *,
                                              const XrHapticBaseHeader *);

static double
now_seconds(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void
shim_log(const char *message)
{
	fprintf(stderr, "[ovrc-shim] %s\n", message);
	fflush(stderr);
}

static void *
load_next_symbol(const char *name)
{
	void *symbol = dlsym(RTLD_NEXT, name);
	if (symbol == NULL) {
		char buffer[256];
		snprintf(buffer, sizeof(buffer), "missing OpenXR symbol %s", name);
		shim_log(buffer);
	}
	return symbol;
}

static void
load_gl_symbols(void)
{
	if (p_glGetTexImage != NULL) {
		return;
	}
	void *lib = dlopen("libGL.so.1", RTLD_LAZY | RTLD_NOLOAD);
	if (lib == NULL) {
		lib = dlopen("libGL.so.1", RTLD_LAZY);
	}
	if (lib == NULL) {
		shim_log("could not open libGL.so.1; texture capture disabled");
		return;
	}
	p_glGetTexImage = (PFN_glGetTexImage)dlsym(lib, "glGetTexImage");
	p_glGetIntegerv = (PFN_glGetIntegerv)dlsym(lib, "glGetIntegerv");
	p_glPixelStorei = (PFN_glPixelStorei)dlsym(lib, "glPixelStorei");
	p_glBindTexture = (PFN_glBindTexture)dlsym(lib, "glBindTexture");
	if (p_glGetTexImage == NULL || p_glGetIntegerv == NULL || p_glPixelStorei == NULL ||
	    p_glBindTexture == NULL) {
		shim_log("libGL is missing texture readback entry points");
		p_glGetTexImage = NULL;
	}
}

static void
read_environment(void)
{
	const char *value;
	if ((value = getenv("OVRC_CAPTURE_HOST")) != NULL && value[0] != '\0') {
		snprintf(g_host, sizeof(g_host), "%s", value);
	}
	if ((value = getenv("OVRC_CAPTURE_PORT")) != NULL) {
		g_port = atoi(value);
	}
	if ((value = getenv("OVRC_CAPTURE_EYE_WIDTH")) != NULL) {
		g_eye_width = atoi(value);
	}
	if ((value = getenv("OVRC_CAPTURE_EYE_HEIGHT")) != NULL) {
		g_eye_height = atoi(value);
	}
	if ((value = getenv("OVRC_CAPTURE_FPS")) != NULL) {
		g_fps = atoi(value);
	}
	if ((value = getenv("OVRC_CAPTURE_QUALITY")) != NULL) {
		g_quality = atoi(value);
	}
	if ((value = getenv("OVRC_CAPTURE_GAMMA")) != NULL) {
		g_gamma = atof(value);
	}
	if ((value = getenv("OVRC_CAPTURE_DEBUG")) != NULL) {
		g_debug = atoi(value) != 0;
	}
	if (g_eye_width < 64) {
		g_eye_width = 64;
	}
	if (g_eye_height < 64) {
		g_eye_height = 64;
	}
	if (g_eye_width > 4096) {
		g_eye_width = 4096;
	}
	if (g_eye_height > 4096) {
		g_eye_height = 4096;
	}
	if (g_fps < 1) {
		g_fps = 1;
	}
	if (g_quality < 20) {
		g_quality = 20;
	}
	if (g_quality > 95) {
		g_quality = 95;
	}
	if (g_gamma < 0.0 || g_gamma > 4.0) {
		g_gamma = 2.2;
	}
	for (int i = 0; i < 256; i++) {
		if (g_gamma > 0.0) {
			double value = 255.0 * pow((double)i / 255.0, 1.0 / g_gamma);
			if (value < 0.0) {
				value = 0.0;
			}
			if (value > 255.0) {
				value = 255.0;
			}
			g_gamma_lut[i] = (uint8_t)(value + 0.5);
		} else {
			g_gamma_lut[i] = (uint8_t)i;
		}
	}
}

static void
enqueue_event_locked(const char *json)
{
	if (g_event_count >= MAX_EVENTS) {
		/* Keep the newest events; drop the oldest. */
		struct event_job *old = g_event_head;
		if (old != NULL) {
			g_event_head = old->next;
			if (g_event_head == NULL) {
				g_event_tail = NULL;
			}
			g_event_count--;
			free(old->json);
			free(old);
		}
	}
	struct event_job *job = (struct event_job *)calloc(1, sizeof(*job));
	if (job == NULL) {
		return;
	}
	job->json = strdup(json);
	if (job->json == NULL) {
		free(job);
		return;
	}
	if (g_event_tail != NULL) {
		g_event_tail->next = job;
	} else {
		g_event_head = job;
	}
	g_event_tail = job;
	g_event_count++;
	pthread_cond_broadcast(&g_cond);
}

static int
connect_to_desktop(void)
{
	char service[16];
	snprintf(service, sizeof(service), "%d", g_port);

	struct addrinfo hints;
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;

	struct addrinfo *result = NULL;
	if (getaddrinfo(g_host, service, &hints, &result) != 0) {
		return -1;
	}
	int fd = -1;
	for (struct addrinfo *entry = result; entry != NULL; entry = entry->ai_next) {
		fd = socket(entry->ai_family, entry->ai_socktype, entry->ai_protocol);
		if (fd < 0) {
			continue;
		}
		if (connect(fd, entry->ai_addr, entry->ai_addrlen) == 0) {
			break;
		}
		close(fd);
		fd = -1;
	}
	freeaddrinfo(result);
	if (fd >= 0) {
		int flags = 1;
		setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flags, sizeof(flags));
	}
	return fd;
}

static int
send_all(int fd, const void *data, size_t size)
{
	const uint8_t *cursor = (const uint8_t *)data;
	while (size > 0) {
		ssize_t written = send(fd, cursor, size, MSG_NOSIGNAL);
		if (written <= 0) {
			return -1;
		}
		cursor += written;
		size -= (size_t)written;
	}
	return 0;
}

static int
send_message(int fd, uint8_t type, const void *payload, uint32_t length)
{
	uint8_t header[FRAME_HEADER_SIZE];
	memcpy(header, FRAME_MAGIC, 4);
	header[4] = type;
	header[5] = 0;
	header[6] = 0;
	header[7] = 0;
	uint32_t wire_length = htole32(length);
	memcpy(header + 8, &wire_length, 4);
	if (send_all(fd, header, sizeof(header)) != 0) {
		return -1;
	}
	if (length > 0 && send_all(fd, payload, length) != 0) {
		return -1;
	}
	return 0;
}

static int
send_event(int fd, const char *json)
{
	return send_message(fd, 2, json, (uint32_t)strlen(json));
}

static int
encode_jpeg(const struct frame_job *frame, uint8_t **out_data, size_t *out_size)
{
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);

	unsigned char *memory = NULL;
	unsigned long memory_size = 0;
	jpeg_mem_dest(&cinfo, &memory, &memory_size);

	cinfo.image_width = frame->width;
	cinfo.image_height = frame->height;
	cinfo.input_components = 3;
	cinfo.in_color_space = JCS_RGB;
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, g_quality, TRUE);
	jpeg_start_compress(&cinfo, TRUE);

	JSAMPROW rows[1];
	while (cinfo.next_scanline < cinfo.image_height) {
		rows[0] = (JSAMPROW)(frame->rgb + (size_t)cinfo.next_scanline * frame->width * 3);
		jpeg_write_scanlines(&cinfo, rows, 1);
	}
	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);

	*out_data = (uint8_t *)memory;
	*out_size = (size_t)memory_size;
	return 0;
}

static int
send_frame(int fd, const struct frame_job *frame)
{
	uint8_t *jpeg = NULL;
	size_t jpeg_size = 0;
	if (encode_jpeg(frame, &jpeg, &jpeg_size) != 0 || jpeg == NULL) {
		return -1;
	}
	if (jpeg_size > 8 * 1024 * 1024) {
		free(jpeg);
		return -1;
	}
	uint32_t payload_length = (uint32_t)(FRAME_PAYLOAD_HEADER + jpeg_size);
	uint8_t *payload = (uint8_t *)malloc(payload_length);
	if (payload == NULL) {
		free(jpeg);
		return -1;
	}
	uint32_t sequence = htole32(frame->sequence);
	uint32_t width = htole32(frame->width);
	uint32_t height = htole32(frame->height);
	uint64_t stamp;
	memcpy(&stamp, &frame->timestamp, sizeof(stamp));
	memcpy(payload, &sequence, 4);
	memcpy(payload + 4, &stamp, 8);
	memcpy(payload + 12, &width, 4);
	memcpy(payload + 16, &height, 4);
	memcpy(payload + FRAME_PAYLOAD_HEADER, jpeg, jpeg_size);
	int result = send_message(fd, 1, payload, payload_length);
	free(payload);
	free(jpeg);
	return result;
}

static void *
sender_main(void *unused)
{
	(void)unused;
	while (1) {
		if (g_socket < 0) {
			g_socket = connect_to_desktop();
			if (g_socket < 0) {
				usleep(500 * 1000);
				continue;
			}
			char ready[256];
			snprintf(ready, sizeof(ready),
			         "{\"type\":\"ready\",\"width\":%d,\"height\":%d,\"fps\":%d}",
			         g_eye_width * 2, g_eye_height, g_fps);
			if (send_event(g_socket, ready) != 0) {
				close(g_socket);
				g_socket = -1;
				continue;
			}
			shim_log("connected to the desktop capture server");
		}

		char *event = NULL;
		struct frame_job frame;
		int have_frame = 0;

		pthread_mutex_lock(&g_lock);
		while (g_event_head == NULL && g_frame_count == 0) {
			pthread_cond_wait(&g_cond, &g_lock);
		}
		if (g_event_head != NULL) {
			struct event_job *job = g_event_head;
			g_event_head = job->next;
			if (g_event_head == NULL) {
				g_event_tail = NULL;
			}
			g_event_count--;
			event = job->json;
			free(job);
		} else {
			frame = g_frames[g_frame_head];
			g_frame_head = (g_frame_head + 1) % MAX_FRAMES;
			g_frame_count--;
			have_frame = 1;
		}
		pthread_mutex_unlock(&g_lock);

		int failed = 0;
		if (event != NULL) {
			failed = send_event(g_socket, event);
			free(event);
		} else if (have_frame) {
			failed = send_frame(g_socket, &frame);
			free(frame.rgb);
		}
		if (failed != 0) {
			close(g_socket);
			g_socket = -1;
			if (have_frame) {
				/* Nothing else to do; the next frame will reconnect. */
			}
		}
	}
	return NULL;
}

static void
initialize_once(void)
{
	if (g_initialized) {
		return;
	}
	g_initialized = 1;
	read_environment();
	g_started_at = now_seconds();

	real_xrCreateSwapchain = (XrResult(*)(XrSession, const XrSwapchainCreateInfo *, XrSwapchain *))
	    load_next_symbol("xrCreateSwapchain");
	real_xrDestroySwapchain = (XrResult(*)(XrSwapchain))load_next_symbol("xrDestroySwapchain");
	real_xrEnumerateSwapchainImages =
	    (XrResult(*)(XrSwapchain, uint32_t, uint32_t *, XrSwapchainImageBaseHeader *))
	        load_next_symbol("xrEnumerateSwapchainImages");
	real_xrAcquireSwapchainImage =
	    (XrResult(*)(XrSwapchain, const XrSwapchainImageAcquireInfo *, uint32_t *))
	        load_next_symbol("xrAcquireSwapchainImage");
	real_xrEndFrame = (XrResult(*)(XrSession, const XrFrameEndInfo *))load_next_symbol("xrEndFrame");
	real_xrCreateAction =
	    (XrResult(*)(XrActionSet, const XrActionCreateInfo *, XrAction *))
	        load_next_symbol("xrCreateAction");
	real_xrStringToPath =
	    (XrResult(*)(XrInstance, const char *, XrPath *))load_next_symbol("xrStringToPath");
	real_xrGetActionStateFloat =
	    (XrResult(*)(XrSession, const XrActionStateGetInfo *, XrActionStateFloat *))
	        load_next_symbol("xrGetActionStateFloat");
	real_xrApplyHapticFeedback =
	    (XrResult(*)(XrSession, const XrHapticActionInfo *, const XrHapticBaseHeader *))
	        load_next_symbol("xrApplyHapticFeedback");

	load_gl_symbols();

	pthread_t thread;
	if (pthread_create(&thread, NULL, sender_main, NULL) == 0) {
		pthread_detach(thread);
	}
	shim_log("OpenXR capture shim active");
}

static struct swapchain_info *
find_swapchain(XrSwapchain handle)
{
	for (int i = 0; i < g_swapchain_count; i++) {
		if (g_swapchains[i].handle == handle) {
			return &g_swapchains[i];
		}
	}
	return NULL;
}

static struct swapchain_info *
add_swapchain(XrSwapchain handle, uint32_t width, uint32_t height)
{
	struct swapchain_info *info = find_swapchain(handle);
	if (info != NULL) {
		info->width = width;
		info->height = height;
		return info;
	}
	if (g_swapchain_count >= MAX_SWAPCHAINS) {
		return NULL;
	}
	info = &g_swapchains[g_swapchain_count++];
	memset(info, 0, sizeof(*info));
	info->handle = handle;
	info->width = width;
	info->height = height;
	return info;
}

static void
remove_swapchain(XrSwapchain handle)
{
	for (int i = 0; i < g_swapchain_count; i++) {
		if (g_swapchains[i].handle != handle) {
			continue;
		}
		free(g_swapchains[i].textures);
		g_swapchains[i] = g_swapchains[g_swapchain_count - 1];
		g_swapchain_count--;
		return;
	}
}

static struct action_info *
find_action(XrAction handle)
{
	for (int i = 0; i < g_action_count; i++) {
		if (g_actions[i].handle == handle) {
			return &g_actions[i];
		}
	}
	return NULL;
}

static const char *
hand_for_path(XrPath path)
{
	if (path == XR_NULL_PATH) {
		return "unknown";
	}
	for (int i = 0; i < g_path_count; i++) {
		if (g_paths[i].path != path) {
			continue;
		}
		if (strstr(g_paths[i].str, "/left") != NULL) {
			return "left";
		}
		if (strstr(g_paths[i].str, "/right") != NULL) {
			return "right";
		}
		return "unknown";
	}
	return "unknown";
}

static int
hand_index(const char *hand)
{
	if (strcmp(hand, "left") == 0) {
		return 0;
	}
	if (strcmp(hand, "right") == 0) {
		return 1;
	}
	return -1;
}

static const XrCompositionLayerProjection *
find_projection_layer(const XrFrameEndInfo *info)
{
	if (info == NULL) {
		return NULL;
	}
	for (uint32_t i = 0; i < info->layerCount; i++) {
		const XrCompositionLayerBaseHeader *layer = info->layers[i];
		if (layer == NULL || layer->type != XR_TYPE_COMPOSITION_LAYER_PROJECTION) {
			continue;
		}
		const XrCompositionLayerProjection *projection =
		    (const XrCompositionLayerProjection *)layer;
		if (projection->viewCount > 0) {
			return projection;
		}
	}
	return NULL;
}

/*
 * Read one swapchain texture into a tightly packed RGB buffer, flipping the GL
 * bottom-left origin and scaling to the configured per-eye size.
 */
static int
read_texture_rgb(GLuint texture,
                 uint32_t source_width,
                 uint32_t source_height,
                 uint8_t *out_rgb,
                 int out_width,
                 int out_height)
{
	if (p_glGetTexImage == NULL) {
		return -1;
	}
	if (source_width == 0 || source_height == 0) {
		return -1;
	}
	size_t source_size = (size_t)source_width * source_height * 4;
	uint8_t *source = (uint8_t *)malloc(source_size);
	if (source == NULL) {
		return -1;
	}

	GLint previous_texture = 0;
	GLint previous_alignment = 4;
	GLint previous_pack_buffer = 0;
	p_glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous_texture);
	p_glGetIntegerv(GL_PACK_ALIGNMENT, &previous_alignment);
	p_glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &previous_pack_buffer);
	if (previous_pack_buffer != 0) {
		/* A pixel pack buffer would redirect glGetTexImage; unbind it. */
		typedef void (*PFN_glBindBuffer)(GLenum, GLuint);
		PFN_glBindBuffer bind_buffer = (PFN_glBindBuffer)dlsym(RTLD_DEFAULT, "glBindBuffer");
		if (bind_buffer != NULL) {
			bind_buffer(GL_PIXEL_PACK_BUFFER, 0);
		}
	}
	p_glPixelStorei(GL_PACK_ALIGNMENT, 1);
	p_glPixelStorei(GL_PACK_ROW_LENGTH, 0);
	p_glPixelStorei(GL_PACK_SKIP_ROWS, 0);
	p_glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
	p_glBindTexture(GL_TEXTURE_2D, texture);
	p_glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, source);

	p_glBindTexture(GL_TEXTURE_2D, (GLuint)previous_texture);
	p_glPixelStorei(GL_PACK_ALIGNMENT, previous_alignment);
	if (previous_pack_buffer != 0) {
		typedef void (*PFN_glBindBuffer)(GLenum, GLuint);
		PFN_glBindBuffer bind_buffer = (PFN_glBindBuffer)dlsym(RTLD_DEFAULT, "glBindBuffer");
		if (bind_buffer != NULL) {
			bind_buffer(GL_PIXEL_PACK_BUFFER, (GLuint)previous_pack_buffer);
		}
	}

	/* Native-size capture is the path used for high-refresh streaming. Avoid
	 * the scaling arithmetic entirely and just flip rows while applying the
	 * gamma lookup table; the JPEG encoder then sees tightly packed RGB. */
	if ((int)source_width == out_width && (int)source_height == out_height) {
		for (int y = 0; y < out_height; y++) {
			const uint8_t *src =
			    source + (size_t)(source_height - 1 - (uint32_t)y) * source_width * 4;
			uint8_t *dst = out_rgb + (size_t)y * out_width * 3;
			for (int x = 0; x < out_width; x++) {
				dst[x * 3 + 0] = g_gamma_lut[src[x * 4 + 0]];
				dst[x * 3 + 1] = g_gamma_lut[src[x * 4 + 1]];
				dst[x * 3 + 2] = g_gamma_lut[src[x * 4 + 2]];
			}
		}
		free(source);
		return 0;
	}

	for (int y = 0; y < out_height; y++) {
		/* GL rows start at the bottom; flip into image order. */
		float sy = ((float)(out_height - 1 - y) + 0.5f) * source_height / out_height - 0.5f;
		if (sy < 0.0f) {
			sy = 0.0f;
		}
		if (sy > source_height - 1.0f) {
			sy = source_height - 1.0f;
		}
		int y0 = (int)sy;
		int y1 = y0 + 1 < (int)source_height ? y0 + 1 : y0;
		float fy = sy - y0;

		uint8_t *row = out_rgb + (size_t)y * out_width * 3;
		for (int x = 0; x < out_width; x++) {
			float sx = ((float)x + 0.5f) * source_width / out_width - 0.5f;
			if (sx < 0.0f) {
				sx = 0.0f;
			}
			if (sx > source_width - 1.0f) {
				sx = source_width - 1.0f;
			}
			int x0 = (int)sx;
			int x1 = x0 + 1 < (int)source_width ? x0 + 1 : x0;
			float fx = sx - x0;

			float rgb[3] = {0.0f, 0.0f, 0.0f};
			for (int dy = 0; dy < 2; dy++) {
				int yy = dy == 0 ? y0 : y1;
				float wy = dy == 0 ? 1.0f - fy : fy;
				for (int dx = 0; dx < 2; dx++) {
					int xx = dx == 0 ? x0 : x1;
					float wx = dx == 0 ? 1.0f - fx : fx;
					float weight = wx * wy;
					const uint8_t *pixel =
					    source + ((size_t)yy * source_width + xx) * 4;
					for (int c = 0; c < 3; c++) {
						rgb[c] += weight * (float)pixel[c];
					}
				}
			}
			for (int c = 0; c < 3; c++) {
				int value = (int)(rgb[c] + 0.5f);
				if (value < 0) {
					value = 0;
				}
				if (value > 255) {
					value = 255;
				}
				row[x * 3 + c] = g_gamma_lut[value];
			}
		}
	}

	free(source);
	return 0;
}

static void
enqueue_frame(uint32_t sequence, double timestamp, int width, int height, uint8_t *rgb)
{
	pthread_mutex_lock(&g_lock);
	if (g_frame_count >= MAX_FRAMES) {
		struct frame_job *oldest = &g_frames[g_frame_head];
		free(oldest->rgb);
		g_frame_head = (g_frame_head + 1) % MAX_FRAMES;
		g_frame_count--;
	}
	int index = (g_frame_head + g_frame_count) % MAX_FRAMES;
	g_frames[index].sequence = sequence;
	g_frames[index].timestamp = timestamp;
	g_frames[index].width = (uint32_t)width;
	g_frames[index].height = (uint32_t)height;
	g_frames[index].rgb = rgb;
	g_frame_count++;
	pthread_cond_broadcast(&g_cond);
	pthread_mutex_unlock(&g_lock);
}

static void
capture_frame(const XrCompositionLayerProjection *projection)
{
	int view_count = (int)projection->viewCount;
	if (view_count > 2) {
		view_count = 2;
	}
	int out_width = g_eye_width * (view_count == 1 ? 1 : 2);
	int out_height = g_eye_height;
	uint8_t *rgb = (uint8_t *)malloc((size_t)out_width * out_height * 3);
	if (rgb == NULL) {
		return;
	}
	memset(rgb, 0, (size_t)out_width * out_height * 3);

	int captured = 0;
	for (int view = 0; view < view_count; view++) {
		XrSwapchain handle = projection->views[view].subImage.swapchain;
		struct swapchain_info *info = find_swapchain(handle);
		if (g_debug) {
			fprintf(stderr,
			        "[ovrc-shim] capture view=%d handle=%llu "
			        "pos=(%.4f %.4f %.4f) rot=(%.4f %.4f %.4f %.4f) "
			        "info=%s valid=%d index=%u count=%u\n",
			        view, (unsigned long long)handle,
			        (double)projection->views[view].pose.position.x,
			        (double)projection->views[view].pose.position.y,
			        (double)projection->views[view].pose.position.z,
			        (double)projection->views[view].pose.orientation.x,
			        (double)projection->views[view].pose.orientation.y,
			        (double)projection->views[view].pose.orientation.z,
			        (double)projection->views[view].pose.orientation.w,
			        info != NULL ? "yes" : "no",
			        info != NULL ? info->last_valid : -1,
			        info != NULL ? info->last_index : 0,
			        info != NULL ? info->count : 0);
		}
		if (info == NULL || !info->last_valid || info->textures == NULL ||
		    info->last_index >= info->count) {
			continue;
		}
		int right = view == 1;
		if (view_count == 2) {
			/* Eye poses share a common head offset, so compare them with
			 * each other instead of testing against the world origin. */
			float center_x = 0.5f * (projection->views[0].pose.position.x +
			                         projection->views[1].pose.position.x);
			if (projection->views[view].pose.position.x < center_x - 1e-6f) {
				right = 0;
			} else if (projection->views[view].pose.position.x > center_x + 1e-6f) {
				right = 1;
			}
		}
		int target_x = right ? g_eye_width : 0;
		uint8_t *target = (uint8_t *)malloc((size_t)g_eye_width * g_eye_height * 3);
		if (target == NULL) {
			continue;
		}
		if (read_texture_rgb(info->textures[info->last_index], info->width, info->height,
		                     target, g_eye_width, g_eye_height) == 0) {
			for (int y = 0; y < g_eye_height; y++) {
				memcpy(rgb + ((size_t)y * out_width + target_x) * 3,
				       target + (size_t)y * g_eye_width * 3,
				       (size_t)g_eye_width * 3);
			}
			captured++;
		}
		free(target);
	}
	if (g_debug) {
		fprintf(stderr, "[ovrc-shim] capture views=%d captured=%d\n", view_count, captured);
	}

	if (captured == 0) {
		free(rgb);
		return;
	}
	if (view_count == 1) {
		/* Duplicate a mono view so downstream SBS handling is uniform. */
		uint8_t *stereo = (uint8_t *)malloc((size_t)g_eye_width * 2 * g_eye_height * 3);
		if (stereo == NULL) {
			free(rgb);
			return;
		}
		for (int y = 0; y < g_eye_height; y++) {
			memcpy(stereo + ((size_t)y * g_eye_width * 2) * 3,
			       rgb + (size_t)y * g_eye_width * 3, (size_t)g_eye_width * 3);
			memcpy(stereo + ((size_t)y * g_eye_width * 2 + g_eye_width) * 3,
			       rgb + (size_t)y * g_eye_width * 3, (size_t)g_eye_width * 3);
		}
		free(rgb);
		rgb = stereo;
		out_width = g_eye_width * 2;
	}

	enqueue_frame(g_sequence++, now_seconds() - g_started_at, out_width, out_height, rgb);
}

/*
 *
 * Interposed OpenXR entry points.
 *
 */

OVRC_EXPORT XrResult
xrCreateSwapchain(XrSession session, const XrSwapchainCreateInfo *create_info, XrSwapchain *swapchain)
{
	pthread_once(&g_once, initialize_once);
	if (real_xrCreateSwapchain == NULL) {
		return XR_ERROR_FUNCTION_UNSUPPORTED;
	}
	XrResult result = real_xrCreateSwapchain(session, create_info, swapchain);
	if (XR_SUCCEEDED(result) && swapchain != NULL && *swapchain != XR_NULL_HANDLE) {
		pthread_mutex_lock(&g_lock);
		add_swapchain(*swapchain, create_info->width, create_info->height);
		pthread_mutex_unlock(&g_lock);
	}
	return result;
}

OVRC_EXPORT XrResult
xrDestroySwapchain(XrSwapchain swapchain)
{
	if (real_xrDestroySwapchain == NULL) {
		return XR_ERROR_FUNCTION_UNSUPPORTED;
	}
	XrResult result = real_xrDestroySwapchain(swapchain);
	pthread_mutex_lock(&g_lock);
	remove_swapchain(swapchain);
	pthread_mutex_unlock(&g_lock);
	return result;
}

OVRC_EXPORT XrResult
xrEnumerateSwapchainImages(XrSwapchain swapchain,
                           uint32_t image_capacity_input,
                           uint32_t *image_count_output,
                           XrSwapchainImageBaseHeader *images)
{
	pthread_once(&g_once, initialize_once);
	if (real_xrEnumerateSwapchainImages == NULL) {
		return XR_ERROR_FUNCTION_UNSUPPORTED;
	}
	XrResult result = real_xrEnumerateSwapchainImages(swapchain, image_capacity_input,
	                                                  image_count_output, images);
	if (XR_SUCCEEDED(result) && images != NULL && image_count_output != NULL) {
		pthread_mutex_lock(&g_lock);
		struct swapchain_info *info = find_swapchain(swapchain);
		if (info != NULL) {
			uint32_t count = *image_count_output;
			GLuint *textures = (GLuint *)calloc(count ? count : 1, sizeof(GLuint));
			if (textures != NULL) {
				for (uint32_t i = 0; i < count; i++) {
					XrSwapchainImageBaseHeader *base = (XrSwapchainImageBaseHeader *)(
					    (uint8_t *)images + i * sizeof(XrSwapchainImageOpenGLKHR));
					if (base->type == XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR) {
						textures[i] =
						    ((XrSwapchainImageOpenGLKHR *)base)->image;
					}
				}
				free(info->textures);
				info->textures = textures;
				info->count = count;
				info->last_valid = 0;
			}
		}
		pthread_mutex_unlock(&g_lock);
	}
	return result;
}

OVRC_EXPORT XrResult
xrAcquireSwapchainImage(XrSwapchain swapchain,
                        const XrSwapchainImageAcquireInfo *acquire_info,
                        uint32_t *index)
{
	if (real_xrAcquireSwapchainImage == NULL) {
		return XR_ERROR_FUNCTION_UNSUPPORTED;
	}
	XrResult result = real_xrAcquireSwapchainImage(swapchain, acquire_info, index);
	if (XR_SUCCEEDED(result) && index != NULL) {
		pthread_mutex_lock(&g_lock);
		struct swapchain_info *info = find_swapchain(swapchain);
		if (info != NULL) {
			info->last_index = *index;
			info->last_valid = 1;
		}
		pthread_mutex_unlock(&g_lock);
	}
	return result;
}

OVRC_EXPORT XrResult
xrEndFrame(XrSession session, const XrFrameEndInfo *frame_end_info)
{
	pthread_once(&g_once, initialize_once);
	if (real_xrEndFrame == NULL) {
		return XR_ERROR_FUNCTION_UNSUPPORTED;
	}

	const XrCompositionLayerProjection *projection = find_projection_layer(frame_end_info);
	double now = now_seconds();
	if (projection != NULL && (g_last_capture <= 0.0 || now - g_last_capture >= 1.0 / g_fps)) {
		g_last_capture = now;
		/* The GL context is current on this thread; capture while the
		 * textures are still guaranteed to be alive. Swapchain metadata is
		 * only mutated on this thread, so no lock is needed here. */
		capture_frame(projection);
	}

	return real_xrEndFrame(session, frame_end_info);
}

OVRC_EXPORT XrResult
xrCreateAction(XrActionSet action_set, const XrActionCreateInfo *create_info, XrAction *action)
{
	pthread_once(&g_once, initialize_once);
	if (real_xrCreateAction == NULL) {
		return XR_ERROR_FUNCTION_UNSUPPORTED;
	}
	XrResult result = real_xrCreateAction(action_set, create_info, action);
	if (XR_SUCCEEDED(result) && action != NULL && create_info != NULL) {
		pthread_mutex_lock(&g_lock);
		if (g_action_count < MAX_ACTIONS) {
			struct action_info *info = &g_actions[g_action_count++];
			info->handle = *action;
			snprintf(info->name, sizeof(info->name), "%s", create_info->actionName);
		}
		pthread_mutex_unlock(&g_lock);
	}
	return result;
}

OVRC_EXPORT XrResult
xrStringToPath(XrInstance instance, const char *path_string, XrPath *path)
{
	pthread_once(&g_once, initialize_once);
	if (real_xrStringToPath == NULL) {
		return XR_ERROR_FUNCTION_UNSUPPORTED;
	}
	XrResult result = real_xrStringToPath(instance, path_string, path);
	if (XR_SUCCEEDED(result) && path != NULL && path_string != NULL) {
		pthread_mutex_lock(&g_lock);
		for (int i = 0; i < g_path_count; i++) {
			if (g_paths[i].path == *path) {
				pthread_mutex_unlock(&g_lock);
				return result;
			}
		}
		if (g_path_count < MAX_PATHS) {
			struct path_info *info = &g_paths[g_path_count++];
			info->path = *path;
			snprintf(info->str, sizeof(info->str), "%s", path_string);
		}
		pthread_mutex_unlock(&g_lock);
	}
	return result;
}

OVRC_EXPORT XrResult
xrGetActionStateFloat(XrSession session,
                      const XrActionStateGetInfo *get_info,
                      XrActionStateFloat *state)
{
	if (real_xrGetActionStateFloat == NULL) {
		return XR_ERROR_FUNCTION_UNSUPPORTED;
	}
	XrResult result = real_xrGetActionStateFloat(session, get_info, state);
	if (XR_SUCCEEDED(result) && get_info != NULL && state != NULL) {
		pthread_mutex_lock(&g_lock);
		struct action_info *action = find_action(get_info->action);
		if (action != NULL && strstr(action->name, "grab") != NULL) {
			const char *hand = hand_for_path(get_info->subactionPath);
			int index = hand_index(hand);
			double now = now_seconds();
			if (index >= 0) {
				float value = state->currentState;
				float previous = g_last_grab_value[index];
				if (previous < 0.0f || fabsf(value - previous) >= 0.05f ||
				    now - g_last_grab[index] >= 0.25) {
					char json[256];
					snprintf(json, sizeof(json),
					         "{\"type\":\"grab\",\"hand\":\"%s\",\"value\":%.3f,"
					         "\"active\":%s}",
					         hand, (double)value,
					         state->isActive ? "true" : "false");
					enqueue_event_locked(json);
					g_last_grab[index] = now;
					g_last_grab_value[index] = value;
				}
			}
		}
		pthread_mutex_unlock(&g_lock);
	}
	return result;
}

OVRC_EXPORT XrResult
xrApplyHapticFeedback(XrSession session,
                      const XrHapticActionInfo *haptic_action_info,
                      const XrHapticBaseHeader *haptic_feedback)
{
	if (real_xrApplyHapticFeedback == NULL) {
		return XR_ERROR_FUNCTION_UNSUPPORTED;
	}
	XrResult result = real_xrApplyHapticFeedback(session, haptic_action_info, haptic_feedback);
	if (XR_SUCCEEDED(result) && haptic_action_info != NULL && haptic_feedback != NULL &&
	    haptic_feedback->type == XR_TYPE_HAPTIC_VIBRATION) {
		const XrHapticVibration *vibration = (const XrHapticVibration *)haptic_feedback;
		pthread_mutex_lock(&g_lock);
		const char *hand = hand_for_path(haptic_action_info->subactionPath);
		int index = hand_index(hand);
		double now = now_seconds();
		if (index >= 0 && (g_last_haptic[index] < 0.0 || now - g_last_haptic[index] >= 0.05)) {
			char json[256];
			snprintf(json, sizeof(json),
			         "{\"type\":\"haptic\",\"hand\":\"%s\",\"amplitude\":%.3f,"
			         "\"duration_ms\":%.3f,\"frequency\":%.2f}",
			         hand, (double)vibration->amplitude,
			         (double)vibration->duration / 1000000.0,
			         (double)vibration->frequency);
			enqueue_event_locked(json);
			g_last_haptic[index] = now;
		}
		pthread_mutex_unlock(&g_lock);
	}
	return result;
}
