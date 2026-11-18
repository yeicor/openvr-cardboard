#include "tracker_device.h"

#include "logging.h"

namespace ovrc {

TrackerDevice::TrackerDevice( std::string serial, int trackerIndex )
	: DeviceBase( std::move( serial ), vr::TrackedDeviceClass_GenericTracker ), trackerIndex_( trackerIndex )
{
}

vr::EVRInitError TrackerDevice::OnActivate( vr::PropertyContainerHandle_t container )
{
	vr::VRProperties()->SetStringProperty( container, vr::Prop_ModelNumber_String, "OpenVR Cardboard Tracker" );
	vr::VRProperties()->SetStringProperty( container, vr::Prop_ControllerType_String, "openvrcardboard_tracker" );
	vr::VRProperties()->SetStringProperty( container, vr::Prop_InputProfilePath_String,
		"{openvrcardboard}/input/tracker/tracker_profile.json" );
	vr::VRProperties()->SetStringProperty( container, vr::Prop_RenderModelName_String, "{htc}vr_tracker_vive_1_0" );
	vr::VRProperties()->SetBoolProperty( container, vr::Prop_DeviceIsWireless_Bool, true );
	vr::VRProperties()->SetBoolProperty( container, vr::Prop_DeviceProvidesBatteryStatus_Bool, false );
	vr::VRProperties()->SetBoolProperty( container, vr::Prop_WillDriftInYaw_Bool, false );
	Log( "tracker %d activated: serial=%s", trackerIndex_, GetSerialNumber().c_str() );
	return vr::VRInitError_None;
}

} // namespace ovrc
