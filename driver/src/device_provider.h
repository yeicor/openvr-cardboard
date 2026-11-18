#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "controller_device.h"
#include "driver_link.h"
#include "hmd_device.h"
#include "openvr_driver.h"
#include "protocol.h"
#include "settings.h"
#include "tracker_device.h"

namespace ovrc {

class DeviceProvider : public vr::IServerTrackedDeviceProvider {
public:
	DeviceProvider() = default;
	~DeviceProvider() = default;

	vr::EVRInitError Init( vr::IVRDriverContext *pDriverContext ) override;
	void Cleanup() override;
	const char *const *GetInterfaceVersions() override;
	void RunFrame() override;
	bool ShouldBlockStandbyMode() override;
	void EnterStandby() override;
	void LeaveStandby() override;

private:
	void OnState( const proto::State &state );
	DeviceBase *FindDevice( uint8_t id );

	DriverSettings settings_;
	std::unique_ptr<DriverLink> link_;
	std::unique_ptr<HmdDevice> hmd_;
	std::unique_ptr<ControllerDevice> leftController_;
	std::unique_ptr<ControllerDevice> rightController_;
	std::vector<std::unique_ptr<TrackerDevice>> trackers_;
	std::vector<DeviceBase *> devices_;

	std::mutex stateMutex_;
	proto::State latestState_{};
	bool hasState_ = false;
	bool stale_ = true;
	double lastStateTime_ = 0.0;
	double lastStatusTime_ = 0.0;
	bool initialized_ = false;
};

} // namespace ovrc
