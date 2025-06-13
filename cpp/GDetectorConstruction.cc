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
#include "G4Tubs.hh"
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
#include "H5Cpp.h"

G4VPhysicalVolume *GDetectorConstruction::Construct() {
    //#include <chrono>
    //auto start = std::chrono::high_resolution_clock::now(); //taking 5 seconds
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


    // Create the world volume
    G4Box* solidWorld = new G4Box("WorldX", worldSizeX / 2, worldSizeY / 2, worldSizeZ / 2);
    G4LogicalVolume* logicWorld = new G4LogicalVolume(solidWorld, worldMaterial, "WorldY");
    logicWorld->SetUserLimits(userLimits2);
    
    G4VPhysicalVolume* physWorld = new G4PVPlacement(0, G4ThreeVector(0, 0, 0), logicWorld, "WorldZ", 0, false, 0, true);
    if (detectorData.isMember("cavern")) {
        const Json::Value caverns = detectorData["cavern"];
        for (const auto& cavern : caverns){
            G4double z_center = cavern["z_center"].asDouble() * m;
            G4double dz = cavern["dz"].asDouble() * m;
            Json::Value cavern_blocks = cavern["components"];
            G4Material* boxMaterial = nist->FindOrBuildMaterial(cavern["material"].asString());
            for (auto block: cavern_blocks){
                std::vector<G4TwoVector> corners;
                for (int i = 0; i < 8; ++i) {
                corners.push_back(G4TwoVector (block[i*2].asDouble() * m, block[i*2+1].asDouble() * m));
                    }
                auto genericV = new G4GenericTrap(G4String("cavern_block"), dz, corners);
                auto logicG = new G4LogicalVolume(genericV, boxMaterial, "cavern_log");
                new G4PVPlacement(0, G4ThreeVector(0, 0, z_center), logicG, "cavern", logicWorld, false, 0, true);
                logicG->SetUserLimits(userLimits2);
            }
        
        }
    }
    if (detectorData.isMember("target")) {
        const Json::Value targets = detectorData["target"];
        int i = 0;
        for (const auto& target : targets) {
            G4double innerRadius = 0;//target["innerRadius"].asDouble() * m;
            G4double outerRadius = target["radius"].asDouble() * m;
            G4double dz = target["dz"].asDouble() * m / 2;
            G4double startAngle = 0*deg;//target["startAngle"].asDouble() * deg;
            G4double spanningAngle = 360*deg;//target["spanningAngle"].asDouble() * deg;
            G4double z_center = target["z_center"].asDouble() * m;
            std::string materialName = target["material"].asString();
            G4Material* cylinderMaterial = nist->FindOrBuildMaterial(materialName);
            G4Tubs* solidCylinder = new G4Tubs("TargetCylinder" + std::to_string(i), innerRadius, outerRadius, dz, startAngle, spanningAngle);
            G4LogicalVolume* logicCylinder = new G4LogicalVolume(solidCylinder, cylinderMaterial, "Cylinder" + std::to_string(i));
            new G4PVPlacement(0, G4ThreeVector(0, 0, z_center), logicCylinder, "Cylinder" + std::to_string(i), logicWorld, false, 0, true);
            logicCylinder->SetUserLimits(userLimits2);
            i++;
        }
    }
    // Process the magnets from the JSON variable
    std::string filename = detectorData["global_field_map"]["B"].asString();
    const Json::Value magnets = detectorData["magnets"];


    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet("B");
    H5::DataSpace dataspace = dataset.getSpace();

    hsize_t dims[2];
    dataspace.getSimpleExtentDims(dims);

    std::vector<double> B_vector(dims[0] * dims[1]);
    dataset.read(B_vector.data(), H5::PredType::NATIVE_DOUBLE);


    G4MagneticField* GlobalmagField = nullptr;
    if (!B_vector.empty()) {
        std::map<std::string, std::vector<double>> ranges;
        std::vector<G4ThreeVector> fields;
        ranges["range_x"] = {detectorData["global_field_map"]["range_x"][0].asDouble() * m, detectorData["global_field_map"]["range_x"][1].asDouble() * m, detectorData["global_field_map"]["range_x"][2].asDouble() * m};
        ranges["range_y"] = {detectorData["global_field_map"]["range_y"][0].asDouble() * m, detectorData["global_field_map"]["range_y"][1].asDouble() * m, detectorData["global_field_map"]["range_y"][2].asDouble() * m};
        ranges["range_z"] = {detectorData["global_field_map"]["range_z"][0].asDouble() * m, detectorData["global_field_map"]["range_z"][1].asDouble() * m, detectorData["global_field_map"]["range_z"][2].asDouble() * m};

        for (size_t i = 0; i < B_vector.size(); i += 3) {
            fields.emplace_back(B_vector[i] * tesla, B_vector[i + 1] * tesla, B_vector[i + 2] * tesla);
        }
        std::vector<double>().swap(B_vector);
        // Determine the interpolation type
        CustomMagneticField::InterpolationType interpType = CustomMagneticField::NEAREST_NEIGHBOR;
        // Define the custom magnetic field
        GlobalmagField = new CustomMagneticField(ranges, fields, interpType);
    }
    //const Json::Value fields = detectorData["field_map"];
    double totalWeight = 0;
    for (const auto& magnet : magnets) {

        std::cout<<"Adding box"<<std::endl;
        // Get the material for the magnet
        std::string materialName = magnet["material"].asString();
        G4Material* boxMaterial = nist->FindOrBuildMaterial(materialName);

        G4double z_center = magnet["z_center"].asDouble() * m;
        G4double dz = magnet["dz"].asDouble() * m;

        Json::Value arb8s = magnet["components"];
        for (auto arb8: arb8s) {
            std::vector<G4TwoVector> corners_two;
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
                std::map<std::string, std::vector<double>> ranges;
                std::vector<G4ThreeVector> fields;
                //const Json::Value& pointsData = field_value[0];
                //const Json::Value& fieldsData = field_value[1];
                ranges["range_x"] = {field_value["range_x"][0].asDouble() * m, field_value["range_x"][1].asDouble() * m, field_value["range_x"][2].asDouble() * m};
                ranges["range_y"] = {field_value["range_y"][0].asDouble() * m, field_value["range_y"][1].asDouble() * m, field_value["range_y"][2].asDouble() * m};
                ranges["range_z"] = {field_value["range_z"][0].asDouble() * m, field_value["range_z"][1].asDouble() * m, field_value["range_z"][2].asDouble() * m};

                const Json::Value& fieldsData = field_value["B"];
                for (Json::ArrayIndex i = 0; i < fieldsData.size(); ++i) {
                    //points.emplace_back(pointsData[i][0].asDouble() * m, pointsData[i][1].asDouble() * m, pointsData[i][2].asDouble() * m);
                    fields.emplace_back(fieldsData[i][0].asDouble() * tesla, fieldsData[i][1].asDouble() * tesla, fieldsData[i][2].asDouble() * tesla);
                }
                // Determine the interpolation type
                CustomMagneticField::InterpolationType interpType = CustomMagneticField::NEAREST_NEIGHBOR;
                // Define the custom magnetic field
                magField = new CustomMagneticField(ranges, fields, interpType);
            }
            
            auto FieldManager = new G4FieldManager();
            FieldManager->SetDetectorField(magField);
            FieldManager->CreateChordFinder(magField);
            auto genericV = new G4GenericTrap(G4String("sdf"), dz, corners_two);
            auto logicG = new G4LogicalVolume(genericV, boxMaterial, "gggvl");
            double volArb = boxMaterial->GetDensity() /(kg/m3)  * genericV->GetCubicVolume()/(m3);
            totalWeight += volArb;
            logicG->SetFieldManager(FieldManager, true);
            new G4PVPlacement(0, G4ThreeVector(0, 0, z_center), logicG, "BoxZ", logicWorld, false, 0, true);
            logicG->SetUserLimits(userLimits2);

        }
    }
    if (GlobalmagField) {
        auto fieldManager = new G4FieldManager();
        fieldManager->SetDetectorField(GlobalmagField);
        fieldManager->CreateChordFinder(GlobalmagField);
        logicWorld->SetFieldManager(fieldManager, true);
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


   return physWorld;
}



GDetectorConstruction::GDetectorConstruction(Json::Value detector_data)
    : detectorData(detector_data) {
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

