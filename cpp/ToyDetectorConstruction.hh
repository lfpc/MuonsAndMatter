//
// Created by Shah Rukh Qasim on 10.07.2024.
//

#ifndef MY_PROJECT_TOYDETECTORCONSTRUCTION_HH
#define MY_PROJECT_TOYDETECTORCONSTRUCTION_HH

#include "DetectorConstruction.hh"
#include "json/json.h"
#include "SlimFilmSensitiveDetector.hh"

class ToyDetectorConstruction : public DetectorConstruction {
public:
    virtual G4VPhysicalVolume *Construct();
public:
    ToyDetectorConstruction(Json::Value detector_data, const std::vector<double>& B_vector);
protected:
    Json::Value detectorData;
    std::vector<double> B_vector;

protected:
    double detectorWeightTotal;
public:
    void setMagneticFieldValue(double strength, double theta, double phi) override;

};

#endif //MY_PROJECT_TOYDETECTORCONSTRUCTION_HH