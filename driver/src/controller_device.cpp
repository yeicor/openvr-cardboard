#include "controller_device.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "logging.h"

namespace ovrc {

namespace {

float ClampUnit( float value )
{
	if ( !std::isfinite( value ) )
		return 0.0f;
	return std::clamp( value, 0.0f, 1.0f );
}

float ClampAxis( float value )
{
	if ( !std::isfinite( value ) )
		return 0.0f;
	return std::clamp( value, -1.0f, 1.0f );
}

const char *RoleName( vr::ETrackedControllerRole role )
{
	return role == vr::TrackedControllerRole_LeftHand ? "left" : "right";
}

} // namespace

ControllerDevice::ControllerDevice( std::string serial, vr::ETrackedControllerRole role, const ControllerProfile &profile )
	: DeviceBase( std::move( serial ), vr::TrackedDeviceClass_Controller ), role_( role ), profile_( profile )
{
	handles_.fill( vr::k_ulInvalidInputComponentHandle );
}

vr::EVRInputError ControllerDevice::CreateBoolean( vr::PropertyContainerHandle_t container, const char *path, Component component )
{
	return vr::VRDriverInput()->CreateBooleanComponent( container, path, &handles_[ component ] );
}

vr::EVRInputError ControllerDevice::CreateScalar( vr::PropertyContainerHandle_t container, const char *path, Component component, bool twoSided )
{
	return vr::VRDriverInput()->CreateScalarComponent( container, path, &handles_[ component ], vr::VRScalarType_Absolute,
		twoSided ? vr::VRScalarUnits_NormalizedTwoSided : vr::VRScalarUnits_NormalizedOneSided );
}

vr::EVRInitError ControllerDevice::OnActivate( vr::PropertyContainerHandle_t container )
{
	const bool left = role_ == vr::TrackedControllerRole_LeftHand;
	const std::string model = profile_.id == "knuckles" ? ( left ? "Knuckles Left" : "Knuckles Right" )
		: profile_.id == "oculus_touch" ? ( left ? "Oculus Touch Left" : "Oculus Touch Right" )
		: "VIVE Controller";

	vr::VRProperties()->SetStringProperty( container, vr::Prop_ModelNumber_String, model.c_str() );
	vr::VRProperties()->SetStringProperty( container, vr::Prop_ControllerType_String, profile_.controllerType.c_str() );
	vr::VRProperties()->SetStringProperty( container, vr::Prop_InputProfilePath_String, profile_.inputProfile.c_str() );
	vr::VRProperties()->SetStringProperty( container, vr::Prop_RenderModelName_String,
		( left ? profile_.renderModelLeft : profile_.renderModelRight ).c_str() );
	vr::VRProperties()->SetInt32Property( container, vr::Prop_ControllerRoleHint_Int32, role_ );
	vr::VRProperties()->SetBoolProperty( container, vr::Prop_DeviceIsWireless_Bool, true );
	vr::VRProperties()->SetBoolProperty( container, vr::Prop_DeviceProvidesBatteryStatus_Bool, false );
	vr::VRProperties()->SetBoolProperty( container, vr::Prop_WillDriftInYaw_Bool, false );

	auto require = [ & ]( vr::EVRInputError error, const char *path ) {
		if ( error != vr::VRInputError_None )
			Log( "controller %s: failed to create input %s (%d)", RoleName( role_ ), path, (int)error );
	};

	if ( profile_.inputs & kInputTrigger ) {
		require( CreateScalar( container, "/input/trigger/value", TriggerValue, false ), "/input/trigger/value" );
		require( CreateBoolean( container, "/input/trigger/click", TriggerClick ), "/input/trigger/click" );
	}
	if ( profile_.inputs & kInputGrip ) {
		require( CreateScalar( container, "/input/grip/value", GripValue, false ), "/input/grip/value" );
		require( CreateBoolean( container, "/input/grip/click", GripClick ), "/input/grip/click" );
		require( CreateBoolean( container, "/input/grip/touch", GripTouch ), "/input/grip/touch" );
	}
	if ( profile_.inputs & kInputJoystick ) {
		require( CreateScalar( container, "/input/joystick/x", JoystickX, true ), "/input/joystick/x" );
		require( CreateScalar( container, "/input/joystick/y", JoystickY, true ), "/input/joystick/y" );
		require( CreateBoolean( container, "/input/joystick/click", JoystickClick ), "/input/joystick/click" );
		require( CreateBoolean( container, "/input/joystick/touch", JoystickTouch ), "/input/joystick/touch" );
	}
	if ( profile_.inputs & kInputTrackpad ) {
		require( CreateScalar( container, "/input/trackpad/x", TrackpadX, true ), "/input/trackpad/x" );
		require( CreateScalar( container, "/input/trackpad/y", TrackpadY, true ), "/input/trackpad/y" );
		require( CreateBoolean( container, "/input/trackpad/click", TrackpadClick ), "/input/trackpad/click" );
		require( CreateBoolean( container, "/input/trackpad/touch", TrackpadTouch ), "/input/trackpad/touch" );
	}
	if ( profile_.inputs & kInputAB ) {
		require( CreateBoolean( container, "/input/a/click", ButtonA ), "/input/a/click" );
		require( CreateBoolean( container, "/input/b/click", ButtonB ), "/input/b/click" );
	}
	if ( profile_.inputs & kInputApplicationMenu )
		require( CreateBoolean( container, "/input/application_menu/click", ApplicationMenu ), "/input/application_menu/click" );
	if ( profile_.inputs & kInputSystem )
		require( CreateBoolean( container, "/input/system/click", System ), "/input/system/click" );
	if ( profile_.inputs & kInputHaptic )
		require( vr::VRDriverInput()->CreateHapticComponent( container, "/output/haptic", &handles_[ Haptic ] ), "/output/haptic" );

	Log( "controller %s activated: profile=%s serial=%s", RoleName( role_ ), profile_.id.c_str(), GetSerialNumber().c_str() );
	return vr::VRInitError_None;
}

void ControllerDevice::UpdateInputs( const proto::ControllerInput &input )
{
	const uint32_t buttons = input.buttons;
	auto setBool = [ & ]( Component component, bool value, double time = 0.0 ) {
		if ( handles_[ component ] != vr::k_ulInvalidInputComponentHandle )
			vr::VRDriverInput()->UpdateBooleanComponent( handles_[ component ], value, time );
	};
	auto setScalar = [ & ]( Component component, float value, double time = 0.0 ) {
		if ( handles_[ component ] != vr::k_ulInvalidInputComponentHandle )
			vr::VRDriverInput()->UpdateScalarComponent( handles_[ component ], value, time );
	};

	setScalar( TriggerValue, ClampUnit( input.trigger ) );
	setBool( TriggerClick, ( buttons & proto::ButtonTriggerClick ) != 0 );
	setScalar( GripValue, ClampUnit( input.grip ) );
	setBool( GripClick, ( buttons & proto::ButtonGripClick ) != 0 );
	setBool( GripTouch, ( buttons & proto::ButtonGripTouch ) != 0 );

	setScalar( JoystickX, ClampAxis( input.joystickX ) );
	setScalar( JoystickY, ClampAxis( input.joystickY ) );
	setBool( JoystickClick, ( buttons & proto::ButtonJoystickClick ) != 0 );
	setBool( JoystickTouch, ( buttons & proto::ButtonJoystickTouch ) != 0 );

	setScalar( TrackpadX, ClampAxis( input.trackpadX ) );
	setScalar( TrackpadY, ClampAxis( input.trackpadY ) );
	setBool( TrackpadClick, ( buttons & proto::ButtonTrackpadClick ) != 0 );
	setBool( TrackpadTouch, ( buttons & proto::ButtonTrackpadTouch ) != 0 );

	setBool( ButtonA, ( buttons & proto::ButtonA ) != 0 );
	setBool( ButtonB, ( buttons & proto::ButtonB ) != 0 );
	setBool( ApplicationMenu, ( buttons & proto::ButtonApplicationMenu ) != 0 );
	setBool( System, ( buttons & proto::ButtonSystem ) != 0 );
}

void ControllerDevice::ProcessEvent( const vr::VREvent_t &event )
{
	if ( event.eventType != vr::VREvent_Input_HapticVibration )
		return;
	if ( handles_[ Haptic ] == vr::k_ulInvalidInputComponentHandle ||
		 event.data.hapticVibration.componentHandle != handles_[ Haptic ] )
		return;

	const uint8_t id = role_ == vr::TrackedControllerRole_LeftHand
		? static_cast<uint8_t>( proto::DeviceId::LeftController )
		: static_cast<uint8_t>( proto::DeviceId::RightController );
	const float amplitude = std::clamp( event.data.hapticVibration.fAmplitude, 0.0f, 1.0f );
	const float frequency = event.data.hapticVibration.fFrequency;
	const float duration = std::max( 0.0f, event.data.hapticVibration.fDurationSeconds );
	if ( hapticCallback_ && amplitude > 0.0f && duration > 0.0f )
		hapticCallback_( id, amplitude, frequency, duration );
}

} // namespace ovrc
