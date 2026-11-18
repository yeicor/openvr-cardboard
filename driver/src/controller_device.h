#pragma once

#include <array>
#include <functional>
#include <string>

#include "device_base.h"
#include "settings.h"

namespace ovrc {

class ControllerDevice : public DeviceBase {
public:
	using HapticCallback = std::function<void( uint8_t deviceId, float amplitude, float frequency, float durationSeconds )>;

	ControllerDevice( std::string serial, vr::ETrackedControllerRole role, const ControllerProfile &profile );
	~ControllerDevice() = default;

	vr::EVRInitError OnActivate( vr::PropertyContainerHandle_t container ) override;
	void UpdateInputs( const proto::ControllerInput &input ) override;
	void ProcessEvent( const vr::VREvent_t &event ) override;
	void SetHapticCallback( HapticCallback callback ) { hapticCallback_ = std::move( callback ); }

	vr::ETrackedControllerRole GetRole() const { return role_; }

private:
	enum Component {
		TriggerClick,
		TriggerValue,
		GripClick,
		GripTouch,
		GripValue,
		JoystickX,
		JoystickY,
		JoystickClick,
		JoystickTouch,
		TrackpadX,
		TrackpadY,
		TrackpadClick,
		TrackpadTouch,
		ButtonA,
		ButtonB,
		ApplicationMenu,
		System,
		Haptic,
		ComponentCount,
	};

	vr::EVRInputError CreateBoolean( vr::PropertyContainerHandle_t container, const char *path, Component component );
	vr::EVRInputError CreateScalar( vr::PropertyContainerHandle_t container, const char *path, Component component, bool twoSided );

	vr::ETrackedControllerRole role_;
	ControllerProfile profile_;
	std::array<vr::VRInputComponentHandle_t, ComponentCount> handles_{};
	HapticCallback hapticCallback_;
};

} // namespace ovrc
