#pragma once

#include <memory>
#include <string>

#include "device_base.h"
#include "settings.h"

namespace ovrc {

class HmdDisplayComponent : public vr::IVRDisplayComponent {
public:
	explicit HmdDisplayComponent( const DisplaySettings &settings ) : settings_( settings ) {}

	void GetWindowBounds( int32_t *pnX, int32_t *pnY, uint32_t *pnWidth, uint32_t *pnHeight ) override;
	bool IsDisplayOnDesktop() override { return settings_.displayOnDesktop; }
	bool IsDisplayRealDisplay() override { return false; }
	void GetRecommendedRenderTargetSize( uint32_t *pnWidth, uint32_t *pnHeight ) override;
	void GetEyeOutputViewport( vr::EVREye eEye, uint32_t *pnX, uint32_t *pnY, uint32_t *pnWidth, uint32_t *pnHeight ) override;
	void GetProjectionRaw( vr::EVREye eEye, float *pfLeft, float *pfRight, float *pfTop, float *pfBottom ) override;
	vr::DistortionCoordinates_t ComputeDistortion( vr::EVREye eEye, float fU, float fV ) override;
	bool ComputeInverseDistortion( vr::HmdVector2_t *pResult, vr::EVREye eEye, uint32_t unChannel, float fU, float fV ) override;

private:
	DisplaySettings settings_;
};

class HmdDevice : public DeviceBase {
public:
	HmdDevice( std::string serial, const DisplaySettings &display, bool headModelPrediction );

	vr::EVRInitError OnActivate( vr::PropertyContainerHandle_t container ) override;
	void *OnGetComponent( const char *componentNameAndVersion ) override;

private:
	DisplaySettings display_;
	bool headModelPrediction_ = true;
	HmdDisplayComponent component_;
};

} // namespace ovrc
