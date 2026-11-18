#pragma once

#include <atomic>
#include <mutex>
#include <string>

#include "openvr_driver.h"
#include "protocol.h"

namespace ovrc {

// Common ITrackedDeviceServerDriver implementation. Poses arrive from the UDP
// thread through ApplyPose(); GetPose() is safe to call from the vrserver
// thread at any time.
class DeviceBase : public vr::ITrackedDeviceServerDriver {
public:
	DeviceBase( std::string serial, vr::ETrackedDeviceClass deviceClass );
	virtual ~DeviceBase() = default;

	vr::EVRInitError Activate( uint32_t unObjectId ) override;
	void EnterStandby() override;
	void *GetComponent( const char *pchComponentNameAndVersion ) override;
	void DebugRequest( const char *pchRequest, char *pchResponseBuffer, uint32_t unResponseBufferSize ) override;
	vr::DriverPose_t GetPose() override;
	void Deactivate() override;

	const std::string &GetSerialNumber() const { return serial_; }
	vr::ETrackedDeviceClass GetDeviceClass() const { return deviceClass_; }
	uint32_t GetDeviceIndex() const { return deviceIndex_.load(); }
	bool IsActive() const { return active_.load(); }

	// Called on the UDP receiver thread.  ``predictionSeconds`` is the horizon
	// the sender already applied; it becomes poseTimeOffset so the runtime does
	// not apply a second prediction on top.
	void ApplyPose( const proto::DevicePose &state, double arrivalTime, double predictionSeconds = 0.0 );
	// Returns true when the device transitioned from tracked to lost.
	bool MarkTrackingLost( double now );
	// Per-device timeout for devices that stop appearing in state packets.
	// Returns true when the device transitioned from tracked to lost.
	bool CheckPoseTimeout( double now, double timeoutSeconds );

	virtual void UpdateInputs( const proto::ControllerInput &input ) { (void)input; }
	virtual void ProcessEvent( const vr::VREvent_t &event ) { (void)event; }

protected:
	virtual vr::EVRInitError OnActivate( vr::PropertyContainerHandle_t container ) = 0;
	virtual void OnDeactivate() {}
	virtual void *OnGetComponent( const char *componentNameAndVersion )
	{
		(void)componentNameAndVersion;
		return nullptr;
	}
	void SetShouldApplyHeadModel( bool value ) { shouldApplyHeadModel_ = value; }

	std::atomic<uint32_t> deviceIndex_{ vr::k_unTrackedDeviceIndexInvalid };
	std::atomic<bool> active_{ false };

private:
	std::string serial_;
	vr::ETrackedDeviceClass deviceClass_;
	mutable std::mutex poseMutex_;
	vr::DriverPose_t pose_{};
	double lastPoseTime_ = 0.0;
	bool trackingLost_ = false;
	bool shouldApplyHeadModel_ = false;
};

} // namespace ovrc
