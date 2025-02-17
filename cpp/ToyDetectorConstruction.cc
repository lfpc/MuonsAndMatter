#include "ToyDetectorConstruction.hh"
#include "DetectorConstruction.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4UserLimits.hh"
#include "G4UniformMagField.hh"
#include "G4ThreeVector.hh"
#include "G4TransportationManager.hh"
#include "G4FieldManager.hh"
#include "G4ChordFinder.hh"
#include "G4MagIntegratorStepper.hh"
#include "G4Mag_UsualEqRhs.hh"
#include "G4ClassicalRK4.hh"
#include "CustomMagneticField.hh"
#include "G4SDManager.hh"
#include <iostream>

// Define the EasyMagneticField class directly in this file
class EasyMagneticField : public G4MagneticField {
public:
    EasyMagneticField() {}
    virtual ~EasyMagneticField() {}

    virtual void GetFieldValue(const G4double Point[4], G4double *Bfield) const override {
        G4double z = Point[2];
        if (z < 10 * m) {
            Bfield[0] = 0.0;
            Bfield[1] = 0.0;
            Bfield[2] = 0.0;
        } else if (z < 20 * m) {
            Bfield[0] = 0.0;
            Bfield[1] = 1.0 * tesla;
            Bfield[2] = 0.0;
        } else if (z < 30 * m) {
            Bfield[0] = 0.0;
            Bfield[1] = -1.0 * tesla;
            Bfield[2] = 0.0;
        } else if (z < 40 * m) {
            Bfield[0] = 1.0 * tesla;
            Bfield[1] = 0.0;
            Bfield[2] = 0.0;
        } else if (z < 50 * m) {
            Bfield[0] = -1.0 * tesla;
            Bfield[1] = 0.0;
            Bfield[2] = 0.0;
        } else if (z < 60 * m) {
            Bfield[0] = 0.0;
            Bfield[1] = -1.0 * tesla;
            Bfield[2] = 0.0;
        } else if (z < 70 * m) {
            Bfield[0] = 0.0;
            Bfield[1] = 1.0 * tesla;
            Bfield[2] = 0.0;
        } else {
            Bfield[0] = 0.0;
            Bfield[1] = 0.0;
            Bfield[2] = 0.0;
        }
    }
};

G4VPhysicalVolume *ToyDetectorConstruction::Construct() {
    double limit_world_time_max_ = 5000 * ns;
    double limit_world_energy_max_ = 100 * eV;

    G4UserLimits* userLimits2 = getLimitsFromDetectorConfig(detectorData);
    std::cout << "Initializing Toy space...\n";

    G4NistManager* nist = G4NistManager::Instance();
    G4Material* worldMaterial = nist->FindOrBuildMaterial("G4_Galactic");

    G4double worldSizeX = detectorData["worldSizeX"].asDouble() * m;
    G4double worldSizeY = detectorData["worldSizeY"].asDouble() * m;
    G4double worldSizeZ = detectorData["worldSizeZ"].asDouble() * m;

    G4Box* solidWorld = new G4Box("WorldX", worldSizeX / 2, worldSizeY / 2, worldSizeZ / 2);
    G4LogicalVolume* logicWorld = new G4LogicalVolume(solidWorld, worldMaterial, "WorldY");
    logicWorld->SetUserLimits(userLimits2);

    G4VPhysicalVolume* physWorld = new G4PVPlacement(0, G4ThreeVector(0, 0, 0), logicWorld, "WorldZ", 0, false, 0, true);

    Json::Value field_value = detectorData["global_field_map"];

    G4MagneticField* GlobalmagField = nullptr;
    if (!B_vector.empty()) {
        if (B_vector.size() == 3) {
            std::cout << "Using uniform magnetic field.\n";
            GlobalmagField = new G4UniformMagField(G4ThreeVector(B_vector[0] * tesla, B_vector[1] * tesla, B_vector[2] * tesla));
        } else {
            std::cout << "Using CustomMagneticField.\n";
            std::map<std::string, std::vector<double>> ranges;
            std::vector<G4ThreeVector> fields;
            ranges["range_x"] = {detectorData["global_field_map"]["range_x"][0].asDouble() * m, detectorData["global_field_map"]["range_x"][1].asDouble() * m, detectorData["global_field_map"]["range_x"][2].asDouble() * m};
            ranges["range_y"] = {detectorData["global_field_map"]["range_y"][0].asDouble() * m, detectorData["global_field_map"]["range_y"][1].asDouble() * m, detectorData["global_field_map"]["range_y"][2].asDouble() * m};
            ranges["range_z"] = {detectorData["global_field_map"]["range_z"][0].asDouble() * m, detectorData["global_field_map"]["range_z"][1].asDouble() * m, detectorData["global_field_map"]["range_z"][2].asDouble() * m};
            for (size_t i = 0; i < B_vector.size(); i += 3) {
                fields.emplace_back(B_vector[i] * tesla, B_vector[i + 1] * tesla, B_vector[i + 2] * tesla);
            }
            CustomMagneticField::InterpolationType interpType = CustomMagneticField::NEAREST_NEIGHBOR;
            GlobalmagField = new CustomMagneticField(ranges, fields, interpType);
        }
    } else {
        std::cout << "No magnetic field vector provided, using EasyMagneticField.\n";
        GlobalmagField = new EasyMagneticField();
    }

    auto fieldManager = new G4FieldManager();
    fieldManager->SetDetectorField(GlobalmagField);
    fieldManager->CreateChordFinder(GlobalmagField);
    logicWorld->SetFieldManager(fieldManager, true);
    std::cout << "Field set...\n";

    return physWorld;
}

ToyDetectorConstruction::ToyDetectorConstruction(Json::Value detector_data, const std::vector<double>& B_vector)
    : detectorData(detector_data), B_vector(B_vector) {
    detectorWeightTotal = 0;
}

void ToyDetectorConstruction::setMagneticFieldValue(double strength, double theta, double phi) {
    std::cout << "cannot set magnetic field value for boxy detector.\n" << std::endl;
}
