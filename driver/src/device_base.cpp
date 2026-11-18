#include "device_base.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "logging.h"

namespace ovrc {

namespace {

void NormalizeQuaternion( vr::HmdQuaternion_t &q )
{
	const double lengthSquared = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
	if ( !std::isfinite( lengthSquared ) || lengthSquared < 1e-8 ) {
		q.x = q.y = q.z = 0.0;
		q.w = 1.0;
		return;
	}
	const double inverseLength = 1.0 / std::sqrt( lengthSquared );
	q.x *= inverseLength;
	q.y *= inverseLength;
	q.z *= inverseLength;
	q.w *= inverseLength;
}

} // namespace

DeviceBase::DeviceBase( std::string serial, vr::ETrackedDeviceClass deviceClass )
	: serial_( std::move( serial ) ), deviceClass_( deviceClass )
{
	pose_.qWorldFromDriverRotation.w = 1.0;
	pose_.qDriverFromHeadRotation.w = 1.0;
	pose_.qRotation.w = 1.0;
	pose_.poseIsValid = false;
	pose_.deviceIsConnected = true;
	pose_.result = vr::TrackingResult_Calibrating_InProgress;
}

vr::EVRInitError DeviceBase::Activate( uint32_t unObjectId )
{
	deviceIndex_ = unObjectId;
	active_ = true;
	vr::PropertyContainerHandle_t container = vr::VRProperties()->TrackedDeviceToPropertyContainer( unObjectId );
	if ( container == vr::k_ulInvalidPropertyContainer ) {
		Log( "device %s: no property container", serial_.c_str() );
		return vr::VRInitError_Driver_Unknown;
	}
	vr::VRProperties()->SetStringProperty( container, vr::Prop_SerialNumber_String, serial_.c_str() );
	vr::VRProperties()->SetStringProperty( container, vr::Prop_TrackingSystemName_String, "openvrcardboard" );
	vr::VRProperties()->SetStringProperty( container, vr::Prop_ManufacturerName_String, "OpenVR Cardboard" );
	return OnActivate( container );
}

void DeviceBase::EnterStandby()
{
}

void *DeviceBase::GetComponent( const char *pchComponentNameAndVersion )
{
	return OnGetComponent( pchComponentNameAndVersion );
}

void DeviceBase::DebugRequest( const char *pchRequest, char *pchResponseBuffer, uint32_t unResponseBufferSize )
{
	(void)pchRequest;
	if ( pchResponseBuffer != nullptr && unResponseBufferSize >= 1 )
		pchResponseBuffer[ 0 ] = '\0';
}

vr::DriverPose_t DeviceBase::GetPose()
{
	std::lock_guard<std::mutex> lock( poseMutex_ );
	return pose_;
}

void DeviceBase::Deactivate()
{
	OnDeactivate();
	active_ = false;
	deviceIndex_ = vr::k_unTrackedDeviceIndexInvalid;
}

void DeviceBase::ApplyPose( const proto::DevicePose &state, double arrivalTime, double predictionSeconds )
{
	vr::DriverPose_t pose{};
	if ( std::isfinite( predictionSeconds ) && predictionSeconds > 0.0 )
		pose.poseTimeOffset = std::min( predictionSeconds, 1.0 );
	else
		pose.poseTimeOffset = 0.0;
	pose.qWorldFromDriverRotation.w = 1.0;
	pose.qDriverFromHeadRotation.w = 1.0;
	pose.qRotation.w = 1.0;
	pose.vecPosition[ 0 ] = pose.vecPosition[ 1 ] = pose.vecPosition[ 2 ] = 0.0f;

	if ( proto::IsFinite3( state.position ) && proto::IsFinite4( state.rotation ) ) {
		for ( int i = 0; i < 3; ++i ) {
			pose.vecPosition[ i ] = state.position[ i ];
			pose.vecVelocity[ i ] = proto::IsFinite3( state.velocity ) ? state.velocity[ i ] : 0.0f;
			pose.vecAngularVelocity[ i ] = proto::IsFinite3( state.angularVelocity ) ? state.angularVelocity[ i ] : 0.0f;
		}
		pose.qRotation.x = state.rotation[ 0 ];
		pose.qRotation.y = state.rotation[ 1 ];
		pose.qRotation.z = state.rotation[ 2 ];
		pose.qRotation.w = state.rotation[ 3 ];
		NormalizeQuaternion( pose.qRotation );
	}

	const bool valid = ( state.flags & proto::DeviceFlags::PoseValid ) != 0;
	const bool connected = ( state.flags & proto::DeviceFlags::Connected ) != 0 || valid;
	const bool orientationOnly = ( state.flags & proto::DeviceFlags::OrientationOnly ) != 0;
	pose.poseIsValid = valid;
	pose.deviceIsConnected = connected;
	pose.shouldApplyHeadModel = shouldApplyHeadModel_ && valid;
	if ( !valid )
		pose.result = vr::TrackingResult_Calibrating_InProgress;
	else if ( orientationOnly )
		pose.result = vr::TrackingResult_Fallback_RotationOnly;
	else
		pose.result = vr::TrackingResult_Running_OK;

	{
		std::lock_guard<std::mutex> lock( poseMutex_ );
		pose_ = pose;
		lastPoseTime_ = arrivalTime;
		trackingLost_ = false;
	}
}

bool DeviceBase::MarkTrackingLost( double now )
{
	std::lock_guard<std::mutex> lock( poseMutex_ );
	if ( trackingLost_ )
		return false;
	trackingLost_ = true;
	lastPoseTime_ = now;
	pose_.poseIsValid = false;
	pose_.result = vr::TrackingResult_Calibrating_InProgress;
	pose_.shouldApplyHeadModel = false;
	pose_.vecVelocity[ 0 ] = pose_.vecVelocity[ 1 ] = pose_.vecVelocity[ 2 ] = 0.0f;
	pose_.vecAngularVelocity[ 0 ] = pose_.vecAngularVelocity[ 1 ] = pose_.vecAngularVelocity[ 2 ] = 0.0f;
	return true;
}

bool DeviceBase::CheckPoseTimeout( double now, double timeoutSeconds )
{
	std::lock_guard<std::mutex> lock( poseMutex_ );
	if ( trackingLost_ || !pose_.poseIsValid )
		return false;
	if ( lastPoseTime_ <= 0.0 || now - lastPoseTime_ <= timeoutSeconds )
		return false;

	trackingLost_ = true;
	pose_.poseIsValid = false;
	pose_.result = vr::TrackingResult_Calibrating_InProgress;
	pose_.shouldApplyHeadModel = false;
	pose_.vecVelocity[ 0 ] = pose_.vecVelocity[ 1 ] = pose_.vecVelocity[ 2 ] = 0.0f;
	pose_.vecAngularVelocity[ 0 ] = pose_.vecAngularVelocity[ 1 ] = pose_.vecAngularVelocity[ 2 ] = 0.0f;
	return true;
}

} // namespace ovrc
