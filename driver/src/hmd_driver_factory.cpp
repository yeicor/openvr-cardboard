#include "device_provider.h"
#include "openvr_driver.h"

#include <cstring>

#if defined( _WIN32 )
#define DRIVER_EXPORT extern "C" __declspec( dllexport )
#elif defined( __GNUC__ ) || defined( __APPLE__ )
#define DRIVER_EXPORT extern "C" __attribute__( ( visibility( "default" ) ) )
#else
#error "Unsupported compiler"
#endif

namespace {
ovrc::DeviceProvider g_deviceProvider;
}

DRIVER_EXPORT void *HmdDriverFactory( const char *pInterfaceName, int *pReturnCode )
{
	if ( pInterfaceName != nullptr && std::strcmp( pInterfaceName, vr::IServerTrackedDeviceProvider_Version ) == 0 )
		return &g_deviceProvider;

	if ( pReturnCode )
		*pReturnCode = vr::VRInitError_Init_InterfaceNotFound;
	return nullptr;
}
