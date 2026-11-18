#include "settings.h"

#include <algorithm>

#include "logging.h"
#include "openvr_driver.h"

namespace ovrc {

namespace {

constexpr const char *kSection = "openvrcardboard";

bool GetBool( const char *key, bool fallback )
{
	vr::EVRSettingsError error = vr::VRSettingsError_None;
	const bool value = vr::VRSettings()->GetBool( kSection, key, &error );
	return error == vr::VRSettingsError_None ? value : fallback;
}

int32_t GetInt( const char *key, int32_t fallback )
{
	vr::EVRSettingsError error = vr::VRSettingsError_None;
	const int32_t value = vr::VRSettings()->GetInt32( kSection, key, &error );
	return error == vr::VRSettingsError_None ? value : fallback;
}

float GetFloat( const char *key, float fallback )
{
	vr::EVRSettingsError error = vr::VRSettingsError_None;
	const float value = vr::VRSettings()->GetFloat( kSection, key, &error );
	return error == vr::VRSettingsError_None ? value : fallback;
}

std::string GetString( const char *key, const std::string &fallback )
{
	char buffer[ 512 ]{};
	vr::EVRSettingsError error = vr::VRSettingsError_None;
	vr::VRSettings()->GetString( kSection, key, buffer, sizeof( buffer ), &error );
	if ( error != vr::VRSettingsError_None || buffer[ 0 ] == '\0' )
		return fallback;
	return buffer;
}

} // namespace

DriverSettings DriverSettings::Load()
{
	DriverSettings settings;
	settings.enableVirtualHmd = GetBool( "enable_virtual_hmd", settings.enableVirtualHmd );
	settings.udpPort = static_cast<uint16_t>( std::clamp( GetInt( "udp_port", settings.udpPort ), 1, 65535 ) );
	settings.poseTimeoutMs = std::clamp( GetInt( "pose_timeout_ms", settings.poseTimeoutMs ), 50, 10000 );
	settings.stateRateHz = static_cast<uint32_t>( std::clamp( GetInt( "state_rate_hz", 250 ), 30, 1000 ) );
	settings.controllerProfile = GetString( "controller_profile", settings.controllerProfile );
	settings.trackerCount = std::clamp( GetInt( "tracker_count", settings.trackerCount ), 0, 16 );
	settings.hmdSerial = GetString( "hmd_serial", settings.hmdSerial );
	settings.leftSerial = GetString( "left_controller_serial", settings.leftSerial );
	settings.rightSerial = GetString( "right_controller_serial", settings.rightSerial );
	settings.trackerSerialPrefix = GetString( "tracker_serial_prefix", settings.trackerSerialPrefix );
	settings.headModelPrediction = GetBool( "head_model_prediction", settings.headModelPrediction );

	settings.display.windowX = GetInt( "display_window_x", settings.display.windowX );
	settings.display.windowY = GetInt( "display_window_y", settings.display.windowY );
	settings.display.windowWidth = std::max( 640, (int)GetInt( "display_window_width", settings.display.windowWidth ) );
	settings.display.windowHeight = std::max( 480, (int)GetInt( "display_window_height", settings.display.windowHeight ) );
	settings.display.renderWidth = std::max( 320, (int)GetInt( "display_render_width", settings.display.renderWidth ) );
	settings.display.renderHeight = std::max( 240, (int)GetInt( "display_render_height", settings.display.renderHeight ) );
	settings.display.frequency = std::clamp( GetFloat( "display_frequency", settings.display.frequency ), 30.0f, 240.0f );
	settings.display.ipdMeters = std::clamp( GetFloat( "ipd_meters", settings.display.ipdMeters ), 0.04f, 0.08f );
	settings.display.displayOnDesktop = GetBool( "display_on_desktop", settings.display.displayOnDesktop );

	if ( settings.display.windowWidth % 2 != 0 )
		++settings.display.windowWidth;

	Log( "settings: hmd=%d controllers=%s trackers=%d udp=%u display=%dx%d render=%dx%d@%.0f",
		settings.enableVirtualHmd ? 1 : 0, settings.controllerProfile.c_str(), settings.trackerCount,
		(unsigned)settings.udpPort, settings.display.windowWidth, settings.display.windowHeight,
		settings.display.renderWidth, settings.display.renderHeight, settings.display.frequency );
	return settings;
}

const ControllerProfile &FindControllerProfile( const std::string &id )
{
	static const ControllerProfile vive{
		"vive_controller",
		"vive_controller",
		"{openvrcardboard}/input/controller_vive.json",
		"vr_controller_vive_1_5",
		"vr_controller_vive_1_5",
		kInputTrigger | kInputGrip | kInputTrackpad | kInputApplicationMenu | kInputSystem | kInputHaptic,
	};
	static const ControllerProfile knuckles{
		"knuckles",
		"knuckles",
		"{openvrcardboard}/input/controller_knuckles.json",
		"{indexcontroller}valve_controller_knu_1_0_left",
		"{indexcontroller}valve_controller_knu_1_0_right",
		kInputTrigger | kInputGrip | kInputJoystick | kInputAB | kInputSystem | kInputHaptic,
	};
	static const ControllerProfile oculus{
		"oculus_touch",
		"oculus_touch",
		"{openvrcardboard}/input/controller_oculus_touch.json",
		"oculus_quest2_controller_left",
		"oculus_quest2_controller_right",
		kInputTrigger | kInputGrip | kInputJoystick | kInputAB | kInputApplicationMenu | kInputSystem | kInputHaptic,
	};
	static const ControllerProfile fallback = vive;

	if ( id == "knuckles" || id == "index" )
		return knuckles;
	if ( id == "oculus_touch" || id == "oculus" || id == "quest" )
		return oculus;
	if ( id == "vive_controller" || id == "vive" )
		return vive;
	return fallback;
}

} // namespace ovrc
