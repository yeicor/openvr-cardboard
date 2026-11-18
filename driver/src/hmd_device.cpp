#include "hmd_device.h"

#include <cstring>

#include "logging.h"

namespace ovrc {

namespace {
constexpr float kProjectionTanX = 1.0f;
constexpr float kProjectionTanY = 0.75f;
} // namespace

void HmdDisplayComponent::GetWindowBounds( int32_t *pnX, int32_t *pnY, uint32_t *pnWidth, uint32_t *pnHeight )
{
	*pnX = settings_.windowX;
	*pnY = settings_.windowY;
	*pnWidth = static_cast<uint32_t>( settings_.windowWidth );
	*pnHeight = static_cast<uint32_t>( settings_.windowHeight );
}

void HmdDisplayComponent::GetRecommendedRenderTargetSize( uint32_t *pnWidth, uint32_t *pnHeight )
{
	*pnWidth = static_cast<uint32_t>( settings_.renderWidth );
	*pnHeight = static_cast<uint32_t>( settings_.renderHeight );
}

void HmdDisplayComponent::GetEyeOutputViewport( vr::EVREye eEye, uint32_t *pnX, uint32_t *pnY, uint32_t *pnWidth, uint32_t *pnHeight )
{
	*pnY = 0;
	*pnWidth = static_cast<uint32_t>( settings_.windowWidth / 2 );
	*pnHeight = static_cast<uint32_t>( settings_.windowHeight );
	*pnX = eEye == vr::Eye_Left ? 0 : static_cast<uint32_t>( settings_.windowWidth / 2 );
}

void HmdDisplayComponent::GetProjectionRaw( vr::EVREye eEye, float *pfLeft, float *pfRight, float *pfTop, float *pfBottom )
{
	(void)eEye;
	*pfLeft = -kProjectionTanX;
	*pfRight = kProjectionTanX;
	*pfTop = kProjectionTanY;
	*pfBottom = -kProjectionTanY;
}

vr::DistortionCoordinates_t HmdDisplayComponent::ComputeDistortion( vr::EVREye eEye, float fU, float fV )
{
	(void)eEye;
	vr::DistortionCoordinates_t coordinates{};
	coordinates.rfBlue[ 0 ] = coordinates.rfGreen[ 0 ] = coordinates.rfRed[ 0 ] = fU;
	coordinates.rfBlue[ 1 ] = coordinates.rfGreen[ 1 ] = coordinates.rfRed[ 1 ] = fV;
	return coordinates;
}

bool HmdDisplayComponent::ComputeInverseDistortion( vr::HmdVector2_t *pResult, vr::EVREye eEye, uint32_t unChannel, float fU, float fV )
{
	(void)pResult;
	(void)eEye;
	(void)unChannel;
	(void)fU;
	(void)fV;
	return false;
}

HmdDevice::HmdDevice( std::string serial, const DisplaySettings &display, bool headModelPrediction )
	: DeviceBase( std::move( serial ), vr::TrackedDeviceClass_HMD ),
	  display_( display ), headModelPrediction_( headModelPrediction ), component_( display )
{
	SetShouldApplyHeadModel( headModelPrediction_ );
}

vr::EVRInitError HmdDevice::OnActivate( vr::PropertyContainerHandle_t container )
{
	vr::VRProperties()->SetStringProperty( container, vr::Prop_ModelNumber_String, "OpenVR Cardboard Virtual HMD" );
	vr::VRProperties()->SetStringProperty( container, vr::Prop_RenderModelName_String, "generic_hmd" );
	vr::VRProperties()->SetBoolProperty( container, vr::Prop_DeviceIsWireless_Bool, true );
	vr::VRProperties()->SetBoolProperty( container, vr::Prop_DeviceProvidesBatteryStatus_Bool, false );
	vr::VRProperties()->SetBoolProperty( container, vr::Prop_HasDisplayComponent_Bool, true );
	vr::VRProperties()->SetBoolProperty( container, vr::Prop_IsOnDesktop_Bool, display_.displayOnDesktop );
	vr::VRProperties()->SetBoolProperty( container, vr::Prop_DisplayDebugMode_Bool, true );
	vr::VRProperties()->SetBoolProperty( container, vr::Prop_ContainsProximitySensor_Bool, false );
	vr::VRProperties()->SetFloatProperty( container, vr::Prop_UserIpdMeters_Float, display_.ipdMeters );
	vr::VRProperties()->SetFloatProperty( container, vr::Prop_DisplayFrequency_Float, display_.frequency );
	vr::VRProperties()->SetFloatProperty( container, vr::Prop_UserHeadToEyeDepthMeters_Float, 0.0f );
	vr::VRProperties()->SetFloatProperty( container, vr::Prop_SecondsFromVsyncToPhotons_Float, 0.011f );
	vr::VRProperties()->SetFloatProperty( container, vr::Prop_SecondsFromPhotonsToVblank_Float, 0.0f );
	Log( "HMD activated: serial=%s window=%dx%d render=%dx%d @ %.0f Hz", GetSerialNumber().c_str(),
		display_.windowWidth, display_.windowHeight, display_.renderWidth, display_.renderHeight, display_.frequency );
	return vr::VRInitError_None;
}

void *HmdDevice::OnGetComponent( const char *componentNameAndVersion )
{
	if ( std::strcmp( componentNameAndVersion, vr::IVRDisplayComponent_Version ) == 0 )
		return &component_;
	return nullptr;
}

} // namespace ovrc
