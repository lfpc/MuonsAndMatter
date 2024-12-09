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
#include "SlimFilmSensitiveDetector.hh"
#include "CustomMagneticField.hh"
//#include "CavernConstruction.hh"

#include <iostream>
#include <G4Trap.hh>
#include <G4GeometryTolerance.hh>

G4VPhysicalVolume *GDetectorConstruction::Construct() {
    double limit_world_time_max_ = 5000 * ns;
    double limit_world_energy_max_ = 100 * eV;

    // Create a user limits object with a maximum step size of 1 mm
//    G4double maxStep = 5 * cm;
//    G4UserLimits* userLimits = new G4UserLimits(maxStep);

    // Define the user limits
    G4double maxTrackLength = DBL_MAX; // No limit on track length
    G4double maxStepLength = DBL_MAX;  // No limit on step length
    maxStepLength = detectorData["max_step_length"].asDouble() * m;
    maxTrackLength = DBL_MAX;
    G4double maxTime = DBL_MAX;        // No limit on time
    G4double minKineticEnergy = 100 * MeV; // Minimum kinetic energy

    // Create an instance of G4UserLimits
    G4UserLimits* userLimits2 = getLimitsFromDetectorConfig(detectorData);
    std::cout<<"Initializing Muon shield design...\n";


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
    logicWorld->SetUserLimits(userLimits2);
//    logicWorld->SetUserLimits(userLimits);
    
    G4VPhysicalVolume* physWorld = new G4PVPlacement(0, G4ThreeVector(worldPositionX, worldPositionY, worldPositionZ), logicWorld, "WorldZ", 0, false, 0, true);
    // Process the magnets from the JSON variable
    const Json::Value magnets = detectorData["magnets"];

    G4MagneticField* GlobalmagField = nullptr;
    //#include <chrono>
    //auto start = std::chrono::high_resolution_clock::now();
    Json::Value field_value = detectorData["global_field_map"]; //taking 5 seconds
    //auto end = std::chrono::high_resolution_clock::now(); //5 seconds until here
    //std::cout<<"TIME: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << std::endl;
    
    if (!field_value.empty()) {
        std::vector<G4ThreeVector> points;
        std::vector<G4ThreeVector> fields;
        const Json::Value& pointsData = field_value[0];
        const Json::Value& fieldsData = field_value[1];
        for (Json::ArrayIndex i = 0; i < pointsData.size(); ++i) {
            points.emplace_back(pointsData[i][0].asDouble() * m, pointsData[i][1].asDouble() * m, pointsData[i][2].asDouble() * m);

            fields.emplace_back(fieldsData[i][0].asDouble() * tesla, fieldsData[i][1].asDouble() * tesla, fieldsData[i][2].asDouble() * tesla);
        }
        // Determine the interpolation type
        CustomMagneticField::InterpolationType interpType = CustomMagneticField::NEAREST_NEIGHBOR;
        // Define the custom magnetic field
        GlobalmagField = new CustomMagneticField(points, fields, interpType);
    }
    //end = std::chrono::high_resolution_clock::now(); //14 seconds until here
    //std::cout<<"TIME" << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << std::endl;
    //const Json::Value fields = detectorData["field_map"];
    double totalWeight = 0;
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
        //G4MagneticField* GlobalmagField = nullptr;
        for (auto arb8: arb8s) {
            std::vector<G4TwoVector> corners_two;
//            G4ThreeVector corners[8];
            Json::Value corners = arb8["corners"];
            
            for (int i = 0; i < 8; ++i) {
                corners_two.push_back(G4TwoVector (corners[i*2].asDouble() * m, corners[i*2+1].asDouble() * m));
            }
            Json::Value field_value = arb8["field"];
            G4double fieldX;
            G4double fieldY;
            G4double fieldZ;
            G4ThreeVector fieldValue;
            G4MagneticField* magField = nullptr;
            if (arb8["field_profile"].asString() == "global") {
                magField = GlobalmagField;
            } else if (arb8["field_profile"].asString() == "uniform"){
                fieldX = field_value[0].asDouble();
                fieldY = field_value[1].asDouble();
                fieldZ = field_value[2].asDouble();
                fieldValue = G4ThreeVector(fieldX * tesla, fieldY * tesla, fieldZ * tesla);
                // Create and set the uniform magnetic field for the box
                magField = new G4UniformMagField(fieldValue);
            } else {
                //std::cout<<"USING Custom field !!!!!\n";
                // Extract magnetic field data from detectorData
                std::vector<G4ThreeVector> points;
                std::vector<G4ThreeVector> fields;
                const Json::Value& pointsData = field_value[0];
                const Json::Value& fieldsData = field_value[1];

                for (Json::ArrayIndex i = 0; i < pointsData.size(); ++i) {
                    points.emplace_back(pointsData[i][0].asDouble() * m, pointsData[i][1].asDouble() * m, pointsData[i][2].asDouble() * m);

                    fields.emplace_back(fieldsData[i][0].asDouble() * tesla, fieldsData[i][1].asDouble() * tesla, fieldsData[i][2].asDouble() * tesla);
                }
                // Determine the interpolation type
                CustomMagneticField::InterpolationType interpType = CustomMagneticField::NEAREST_NEIGHBOR;
                if (arb8["field_profile"].asString() == "linear_interpolation") {
                    interpType = CustomMagneticField::LINEAR;
                }
                // Define the custom magnetic field
                
                magField = new CustomMagneticField(points, fields, interpType);
            }
            auto FieldManager = new G4FieldManager();
            FieldManager->SetDetectorField(magField);
            FieldManager->CreateChordFinder(magField);
            // Continue with your existing code
            auto genericV = new G4GenericTrap(G4String("sdf"), dz, corners_two);
            auto logicG = new G4LogicalVolume(genericV, boxMaterial, "gggvl");
            double volArb = boxMaterial->GetDensity() /(kg/m3)  * genericV->GetCubicVolume()/(m3);
            totalWeight += volArb;
            logicG->SetFieldManager(FieldManager, true);
            new G4PVPlacement(0, G4ThreeVector(0, 0, z_center), logicG, "BoxZ", logicWorld, false, 0, true);
            logicG->SetUserLimits(userLimits2);


//            break;
        }
    }




    sensitiveLogical = nullptr;
    if (detectorData.isMember("sensitive_film")) {
        G4Material* air = nist->FindOrBuildMaterial("G4_AIR");

        const Json::Value sensitiveFilm = detectorData["sensitive_film"];
        double dz = sensitiveFilm["dz"].asDouble() * m;
        double dx = sensitiveFilm["dx"].asDouble() * m;
        double dy = sensitiveFilm["dy"].asDouble() * m;
        double z_center = sensitiveFilm["z_center"].asDouble() * m;

        auto sensitiveBox = new G4Box("sensitive_film", dx/2, dy/2, dz/2);
        sensitiveLogical = new G4LogicalVolume(sensitiveBox, air, "sensitive_film_logic");
        new G4PVPlacement(0, G4ThreeVector(0, 0, z_center), sensitiveLogical, "sensitive_plc", logicWorld, false, 0, true);
        sensitiveLogical->SetUserLimits(userLimits2);

        std::cout<<"Sensitive film placed at the end.\n";
    }
    else {
        std::cout<<"Sensitive film skipped.\n";
    }



    detectorWeightTotal = totalWeight;
    //auto Cavern = ConstructShapes(concrete, logicWorld, z_transition, zEndOfAbsorb);


    // Return the physical world
    return physWorld;
}



GDetectorConstruction::GDetectorConstruction(Json::Value detector_data) {
    detectorData = detector_data;
    detectorWeightTotal = 0;
}

void GDetectorConstruction::setMagneticFieldValue(double strength, double theta, double phi) {
//    DetectorConstruction::setMagneticFieldValue(strength, theta, phi);
std::cout<<"cannot set magnetic field value for boxy detector.\n"<<std::endl;
}

double GDetectorConstruction::getDetectorWeight() {
    return detectorWeightTotal;
}

void GDetectorConstruction::ConstructSDandField() {
    G4VUserDetectorConstruction::ConstructSDandField();

    // Attach the sensitive detector to the logical volume
    if (sensitiveLogical) {
        auto* sdManager = G4SDManager::GetSDMpointer();

        G4String sdName = "MySensitiveDetector";
        slimFilmSensitiveDetector = new SlimFilmSensitiveDetector(sdName);
        sdManager->AddNewDetector(slimFilmSensitiveDetector);
        sensitiveLogical->SetSensitiveDetector(slimFilmSensitiveDetector);
        std::cout<<"Sensitive set...\n";
    }

}

