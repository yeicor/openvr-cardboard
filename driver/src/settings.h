#pragma once

#include <cstdint>
#include <string>

namespace ovrc {

enum ControllerInputSupport : uint32_t {
	kInputTrigger = 1u << 0,
	kInputGrip = 1u << 1,
	kInputJoystick = 1u << 2,
	kInputTrackpad = 1u << 3,
	kInputAB = 1u << 4,
	kInputApplicationMenu = 1u << 5,
	kInputSystem = 1u << 6,
	kInputHaptic = 1u << 7,
};

struct ControllerProfile {
	std::string id;
	std::string controllerType;
	std::string inputProfile;
	std::string renderModelLeft;
	std::string renderModelRight;
	uint32_t inputs = 0;
};

struct DisplaySettings {
	int windowX = 0;
	int windowY = 0;
	int windowWidth = 1920;
	int windowHeight = 1080;
	int renderWidth = 1080;
	int renderHeight = 1200;
	float frequency = 90.0f;
	float ipdMeters = 0.063f;
	bool displayOnDesktop = true;
};

struct DriverSettings {
	bool enableVirtualHmd = true;
	uint16_t udpPort = 39100;
	int poseTimeoutMs = 1000;
	uint32_t stateRateHz = 250;
	std::string controllerProfile = "vive_controller";
	int trackerCount = 0;
	std::string hmdSerial = "OVRC-HMD-0001";
	std::string leftSerial = "OVRC-CTRL-LEFT";
	std::string rightSerial = "OVRC-CTRL-RIGHT";
	std::string trackerSerialPrefix = "OVRC-TRK-";
	bool headModelPrediction = true;
	DisplaySettings display;

	static DriverSettings Load();
};

const ControllerProfile &FindControllerProfile( const std::string &id );

} // namespace ovrc
