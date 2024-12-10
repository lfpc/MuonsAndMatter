#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include <G4UniformMagField.hh>
#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"
#include "json/json.h"
#include "G4UserLimits.hh"

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
    DetectorConstruction();
    DetectorConstruction(Json::Value detectorData);
    virtual ~DetectorConstruction();

    virtual G4VPhysicalVolume* Construct();
    virtual void setMagneticFieldValue(double strength, double theta, double phi);
    virtual G4UserLimits * getLimitsFromDetectorConfig(const Json::Value& detectorData);
    virtual double getDetectorWeight();
protected:
    G4UniformMagField* magField;
    Json::Value detectorData;
};

#endif