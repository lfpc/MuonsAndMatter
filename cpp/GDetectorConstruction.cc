//
// Created by Shah Rukh Qasim on 10.07.2024.
//

#include "GDetectorConstruction.hh"
#include "DetectorConstruction.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4Sphere.hh"
#include "G4UserLimits.hh"
#include "G4UniformMagField.hh"
#include "G4ThreeVector.hh"
#include "G4ThreeVector.hh"
#include "G4TransportationManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4Sphere.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4UniformMagField.hh"
#include "G4UserLimits.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VisAttributes.hh"
#include "G4FieldManager.hh"
#include "G4TransportationManager.hh"
#include "G4ChordFinder.hh"
#include "G4MagIntegratorStepper.hh"
#include "G4Mag_UsualEqRhs.hh"
#include "G4PropagatorInField.hh"
#include "G4ClassicalRK4.hh"
#include "G4Para.hh"
#include "G4GenericTrap.hh"


#include <iostream>
#include <G4Trap.hh>
#include <G4GeometryTolerance.hh>

G4VPhysicalVolume *GDetectorConstruction::Construct() {
    double limit_world_time_max_ = 5000 * ns;
    double limit_world_energy_max_ = 100 * eV;

    // Create a user limits object with a maximum step size of 1 mm
    G4double maxStep = 5 * cm;
    G4UserLimits* userLimits = new G4UserLimits(maxStep);

    // Get NIST material manager
    G4NistManager* nist = G4NistManager::Instance();

    // Define the world material
    G4Material* worldMaterial = nist->FindOrBuildMaterial("G4_AIR");
    // Get the world size from the JSON variable
    G4double worldSizeX = detectorData["worldSizeX"].asDouble() * m;
    G4double worldSizeY = detectorData["worldSizeY"].asDouble() * m;
    G4double worldSizeZ = detectorData["worldSizeZ"].asDouble() * m;

    G4double worldPositionX = detectorData["worldPositionX"].asDouble() * m;
    G4double worldPositionY = detectorData["worldPositionY"].asDouble() * m;
    G4double worldPositionZ = detectorData["worldPositionZ"].asDouble() * m;

    // Create the world volume
    G4Box* solidWorld = new G4Box("WorldX", worldSizeX / 2, worldSizeY / 2, worldSizeZ / 2);
    G4LogicalVolume* logicWorld = new G4LogicalVolume(solidWorld, worldMaterial, "WorldY");
    logicWorld->SetUserLimits(userLimits);

    G4VPhysicalVolume* physWorld = new G4PVPlacement(0, G4ThreeVector(worldPositionX, worldPositionY, worldPositionZ), logicWorld, "WorldZ", 0, false, 0, true);

    // Process the magnets from the JSON variable
    const Json::Value magnets = detectorData["magnets"];
    for (const auto& magnet : magnets) {

        std::cout<<"Adding box"<<std::endl;
        // Get the material for the magnet
        std::string materialName = magnet["material"].asString();
        G4Material* boxMaterial = nist->FindOrBuildMaterial(materialName);

//        // Get the dimensions of the box
//        G4double boxSizeX = magnet["sizeX"].asDouble() * m;
//        G4double boxSizeY = magnet["sizeY"].asDouble() * m;
//        G4double boxSizeZ = magnet["sizeZ"].asDouble() * m;
//
//        // Get the position of the box
//        G4double posX = magnet["posX"].asDouble() * m;
//        G4double posY = magnet["posY"].asDouble() * m;
//        G4double posZ = magnet["posZ"].asDouble() * m;
        G4double z_center = magnet["z_center"].asDouble() * m;
        G4double dz = magnet["dz"].asDouble() * m;


        Json::Value arb8s = magnet["components"];
        for (auto arb8: arb8s) {
            std::vector<G4TwoVector> corners_two;
//            G4ThreeVector corners[8];

            Json::Value corners = arb8["corners"];

            for (int i = 0; i < 8; ++i) {
                corners_two.push_back(G4TwoVector (corners[i*2].asDouble() * m, corners[i*2+1].asDouble() * m));
            }

            Json::Value field_value = arb8["field"];

            G4double fieldX = field_value[0].asDouble();
            G4double fieldY = field_value[1].asDouble();
            G4double fieldZ = field_value[2].asDouble();
            G4ThreeVector fieldValue = G4ThreeVector(fieldX * tesla, fieldY * tesla, fieldZ * tesla);

            // Create and set the magnetic field for the box
            auto thingMagField = new G4UniformMagField(fieldValue);
            auto thingFieldManager = new G4FieldManager();
            thingFieldManager->SetDetectorField(thingMagField);

            // Create the equation of motion and the stepper for the thing
    //        G4Mag_UsualEqRhs* equationOfMotion = new G4Mag_UsualEqRhs(thingMagField);
    //        G4MagIntegratorStepper* stepper = new G4ClassicalRK4(equationOfMotion);

            // Create the chord finder for the thing
            thingFieldManager->CreateChordFinder(thingMagField);

//            logicBox->SetFieldManager(boxFieldManager, true);



            // Continue with your existing code
            auto genericV = new G4GenericTrap(G4String("sdf"), dz, corners_two);
            auto logicG = new G4LogicalVolume(genericV, boxMaterial, "gggvl");
            logicG->SetFieldManager(thingFieldManager, false);
            new G4PVPlacement(0, G4ThreeVector(0, 0, z_center), logicG, "BoxZ", logicWorld, false, 0, true);

//            break;
        }
//        break;


//        // Get the magnetic field vector for the box
//        G4double fieldX = magnet["fieldX"].asDouble();
//        G4double fieldY = magnet["fieldY"].asDouble();
//        G4double fieldZ = magnet["fieldZ"].asDouble();
//        G4ThreeVector fieldValue = G4ThreeVector(fieldX * tesla, fieldY * tesla, fieldZ * tesla);
//
//        // Create and set the magnetic field for the box
//        G4UniformMagField* boxMagField = new G4UniformMagField(fieldValue);
//        G4FieldManager* boxFieldManager = new G4FieldManager();
//        boxFieldManager->SetDetectorField(boxMagField);
//
//        // Create the equation of motion and the stepper for the box
////        G4Mag_UsualEqRhs* equationOfMotion = new G4Mag_UsualEqRhs(boxMagField);
////        G4MagIntegratorStepper* stepper = new G4ClassicalRK4(equationOfMotion);
//
//        // Create the chord finder for the box
//        boxFieldManager->CreateChordFinder(boxMagField);
//
//        logicBox->SetFieldManager(boxFieldManager, true);
    }

    // Return the physical world
    return physWorld;
}



GDetectorConstruction::GDetectorConstruction(std::string detector_data) {
    Json::CharReaderBuilder readerBuilder;
    std::string errs;

    std::istringstream iss(detector_data);
    if (Json::parseFromStream(readerBuilder, iss, &detectorData, &errs)) {
        // Output the parsed JSON object
        std::cout << detectorData["worldSizeX"] << std::endl;
    } else {
        std::cerr << "Failed to parse JSON: " << errs << std::endl;
    }

}

void GDetectorConstruction::setMagneticFieldValue(double strength, double theta, double phi) {
//    DetectorConstruction::setMagneticFieldValue(strength, theta, phi);
std::cout<<"cannot set magnetic field value for boxy detector.\n"<<std::endl;
}
