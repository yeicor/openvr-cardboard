#include "device_provider.h"

#include <algorithm>
#include <cstring>

#include "logging.h"
#include "time_util.h"

namespace ovrc {

namespace {
constexpr uint32_t kStatusIntervalMs = 1000;
}

vr::EVRInitError DeviceProvider::Init( vr::IVRDriverContext *pDriverContext )
{
	VR_INIT_SERVER_DRIVER_CONTEXT( pDriverContext );
	Log( "OpenVR Cardboard driver starting" );

	settings_ = DriverSettings::Load();
	link_ = std::make_unique<DriverLink>();

	if ( settings_.enableVirtualHmd ) {
		hmd_ = std::make_unique<HmdDevice>( settings_.hmdSerial, settings_.display, settings_.headModelPrediction );
		if ( !vr::VRServerDriverHost()->TrackedDeviceAdded( hmd_->GetSerialNumber().c_str(), vr::TrackedDeviceClass_HMD, hmd_.get() ) ) {
			Log( "failed to add virtual HMD" );
			return vr::VRInitError_Driver_Unknown;
		}
		devices_.push_back( hmd_.get() );
	}

	const ControllerProfile &profile = FindControllerProfile( settings_.controllerProfile );
	leftController_ = std::make_unique<ControllerDevice>( settings_.leftSerial, vr::TrackedControllerRole_LeftHand, profile );
	rightController_ = std::make_unique<ControllerDevice>( settings_.rightSerial, vr::TrackedControllerRole_RightHand, profile );

	const auto hapticCallback = [ this ]( uint8_t deviceId, float amplitude, float frequency, float duration ) {
		if ( link_ && !link_->SendHaptic( deviceId, amplitude, frequency, duration ) )
			Log( "unable to forward haptic event for device %u", (unsigned)deviceId );
	};
	leftController_->SetHapticCallback( hapticCallback );
	rightController_->SetHapticCallback( hapticCallback );

	for ( auto *controller : { leftController_.get(), rightController_.get() } ) {
		if ( !vr::VRServerDriverHost()->TrackedDeviceAdded( controller->GetSerialNumber().c_str(),
				 vr::TrackedDeviceClass_Controller, controller ) ) {
			Log( "failed to add controller %s", controller->GetSerialNumber().c_str() );
			return vr::VRInitError_Driver_Unknown;
		}
		devices_.push_back( controller );
	}

	for ( int i = 0; i < settings_.trackerCount; ++i ) {
		const std::string serial = settings_.trackerSerialPrefix + std::to_string( i );
		auto tracker = std::make_unique<TrackerDevice>( serial, i );
		if ( !vr::VRServerDriverHost()->TrackedDeviceAdded( serial.c_str(), vr::TrackedDeviceClass_GenericTracker, tracker.get() ) ) {
			Log( "failed to add tracker %s", serial.c_str() );
			return vr::VRInitError_Driver_Unknown;
		}
		devices_.push_back( tracker.get() );
		trackers_.push_back( std::move( tracker ) );
	}

	std::string error;
	if ( !link_->Start( settings_.udpPort, [ this ]( const proto::State &state ) { OnState( state ); }, &error ) ) {
		Log( "failed to start driver link: %s", error.c_str() );
		return vr::VRInitError_Driver_Unknown;
	}

	initialized_ = true;
	lastStatusTime_ = MonotonicSeconds();
	Log( "OpenVR Cardboard driver ready: %zu devices", devices_.size() );
	return vr::VRInitError_None;
}

void DeviceProvider::Cleanup()
{
	Log( "OpenVR Cardboard driver shutting down" );
	if ( link_ )
		link_->Stop();
	devices_.clear();
	trackers_.clear();
	leftController_.reset();
	rightController_.reset();
	hmd_.reset();
	link_.reset();
	initialized_ = false;
	VR_CLEANUP_SERVER_DRIVER_CONTEXT();
}

const char *const *DeviceProvider::GetInterfaceVersions()
{
	return vr::k_InterfaceVersions;
}

void DeviceProvider::RunFrame()
{
	if ( !initialized_ )
		return;

	const double now = MonotonicSeconds();
	proto::ControllerInput leftInput{};
	proto::ControllerInput rightInput{};
	leftInput.id = static_cast<uint8_t>( proto::DeviceId::LeftController );
	rightInput.id = static_cast<uint8_t>( proto::DeviceId::RightController );
	bool markLost = false;
	bool haveState = false;
	bool stale = false;

	{
		std::lock_guard<std::mutex> lock( stateMutex_ );
		haveState = hasState_;
		stale = stale_;
		if ( hasState_ ) {
			for ( uint8_t i = 0; i < latestState_.inputCount; ++i ) {
				const auto &input = latestState_.inputs[ i ];
				if ( input.id == leftInput.id )
					leftInput = input;
				else if ( input.id == rightInput.id )
					rightInput = input;
			}
			if ( !stale_ && now - lastStateTime_ > settings_.poseTimeoutMs / 1000.0 ) {
				stale_ = true;
				stale = true;
				markLost = true;
			}
		}
	}

	if ( markLost ) {
		Log( "desktop state timed out after %d ms", settings_.poseTimeoutMs );
		for ( auto *device : devices_ )
			device->MarkTrackingLost( now );
	} else {
		// A device that stops appearing in otherwise healthy state packets
		// must still become tracking-lost after the configured timeout.
		const double timeoutSeconds = settings_.poseTimeoutMs / 1000.0;
		for ( auto *device : devices_ ) {
			if ( device->CheckPoseTimeout( now, timeoutSeconds ) )
				Log( "device %s tracking timed out", device->GetSerialNumber().c_str() );
		}
	}

	if ( !haveState || stale ) {
		leftInput = proto::ControllerInput{};
		rightInput = proto::ControllerInput{};
		leftInput.id = static_cast<uint8_t>( proto::DeviceId::LeftController );
		rightInput.id = static_cast<uint8_t>( proto::DeviceId::RightController );
	}

	leftController_->UpdateInputs( leftInput );
	rightController_->UpdateInputs( rightInput );

	vr::VREvent_t event{};
	while ( vr::VRServerDriverHost()->PollNextEvent( &event, sizeof( event ) ) ) {
		for ( auto *device : devices_ )
			device->ProcessEvent( event );
	}

	if ( now - lastStatusTime_ >= kStatusIntervalMs / 1000.0 ) {
		lastStatusTime_ = now;
		if ( link_ && link_->HasClient() )
			link_->SendStatus( static_cast<uint32_t>( devices_.size() ) );
	}
}

bool DeviceProvider::ShouldBlockStandbyMode()
{
	return false;
}

void DeviceProvider::EnterStandby()
{
	Log( "driver entering standby" );
}

void DeviceProvider::LeaveStandby()
{
	Log( "driver leaving standby" );
}

DeviceBase *DeviceProvider::FindDevice( uint8_t id )
{
	switch ( id ) {
	case static_cast<uint8_t>( proto::DeviceId::Hmd ):
		return hmd_.get();
	case static_cast<uint8_t>( proto::DeviceId::LeftController ):
		return leftController_.get();
	case static_cast<uint8_t>( proto::DeviceId::RightController ):
		return rightController_.get();
	default:
		break;
	}
	if ( id >= static_cast<uint8_t>( proto::DeviceId::TrackerBase ) ) {
		const std::size_t index = id - static_cast<uint8_t>( proto::DeviceId::TrackerBase );
		if ( index < trackers_.size() )
			return trackers_[ index ].get();
	}
	return nullptr;
}

void DeviceProvider::OnState( const proto::State &state )
{
	const double now = MonotonicSeconds();
	{
		std::lock_guard<std::mutex> lock( stateMutex_ );
		latestState_ = state;
		hasState_ = true;
		stale_ = false;
		lastStateTime_ = now;
	}

	const double predictionSeconds = static_cast<double>( state.predictionMs ) * 0.001;
	for ( uint8_t i = 0; i < state.deviceCount; ++i ) {
		const auto &pose = state.devices[ i ];
		if ( DeviceBase *device = FindDevice( pose.id ) )
			device->ApplyPose( pose, now, predictionSeconds );
	}
}

} // namespace ovrc
