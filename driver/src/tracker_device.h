#pragma once

#include <string>

#include "device_base.h"

namespace ovrc {

class TrackerDevice : public DeviceBase {
public:
	TrackerDevice( std::string serial, int trackerIndex );

	vr::EVRInitError OnActivate( vr::PropertyContainerHandle_t container ) override;

	int GetTrackerIndex() const { return trackerIndex_; }

private:
	int trackerIndex_ = 0;
};

} // namespace ovrc
