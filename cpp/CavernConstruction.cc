/*#include "CavernConstruction.hh"

G4LogicalVolume* ConstructShapes(G4Material* concrete, G4LogicalVolume* logicWorld, double z_transition, double zEndOfAbsorb) {
    // Define parameters for the shapes
    G4double TCC8_length = 170 * m;
    G4double ECN3_length = 100 * m;
    G4double TCC8_trench_length = 12 * m;
    G4double rock_x = 20 * m;
    G4double rock_y = 20 * m;
    G4double rock_z = TCC8_length / 2. + ECN3_length / 2. + 5 * m;
    G4double muon_shield_cavern_x = 5 * m;
    G4double muon_shield_cavern_y = 3.75 * m;
    G4double muon_shield_cavern_z = TCC8_length / 2.;
    G4double TCC8_shift_x = 2.3 * m;
    G4double TCC8_shift_y = 1.75 * m;
    G4double TCC8_shift_z = -TCC8_length / 2.;
    G4double experiment_rock_x = 20 * m;
    G4double experiment_rock_y = 20 * m;
    G4double experiment_rock_z = ECN3_length / 2.;
    G4double experiment_cavern_x = 8 * m;
    G4double experiment_cavern_y = 7.5 * m;
    G4double experiment_cavern_z = ECN3_length / 2.;
    G4double ECN3_shift_x = 3.5 * m;
    G4double ECN3_shift_y = 4 * m;
    G4double ECN3_shift_z = ECN3_length / 2.;
    G4double yoke_pit_x = 3.5 * m;
    G4double yoke_pit_y = 4.3 * m + 1 * cm;
    G4double yoke_pit_z = 2.5 * m;
    G4double yoke_pit_shift_x = 0 * m;
    G4double yoke_pit_shift_y = 0 * m;
    G4double yoke_pit_shift_z = 31 * m - z_transition;
    G4double target_pit_x = 2 * m;
    G4double target_pit_y = 0.5 * m;
    G4double target_pit_z = 2 * m;
    G4double target_pit_shift_x = 0 * m;
    G4double target_pit_shift_y = -2.5 * m;
    G4double target_pit_shift_z = zEndOfAbsorb - 2 * m - z_transition;

    // Create the shapes using the parameters
    auto rock = new G4Box("rock", rock_x, rock_y, rock_z);
    auto muon_shield_cavern = new G4Box("muon_shield_cavern", muon_shield_cavern_x, muon_shield_cavern_y, muon_shield_cavern_z);
    auto experiment_rock = new G4Box("experiment_rock", experiment_rock_x, experiment_rock_y, experiment_rock_z);
    auto experiment_cavern = new G4Box("experiment_cavern", experiment_cavern_x, experiment_cavern_y, experiment_cavern_z);
    auto yoke_pit = new G4Box("yoke_pit", yoke_pit_x, yoke_pit_y, yoke_pit_z);
    auto target_pit = new G4Box("target_pit", target_pit_x, target_pit_y, target_pit_z);

    // Create the logical volumes
    auto rock_log = new G4LogicalVolume(rock, concrete, "rock_log");
    auto muon_shield_cavern_log = new G4LogicalVolume(muon_shield_cavern, concrete, "muon_shield_cavern_log");
    auto experiment_rock_log = new G4LogicalVolume(experiment_rock, concrete, "experiment_rock_log");
    auto experiment_cavern_log = new G4LogicalVolume(experiment_cavern, concrete, "experiment_cavern_log");
    auto yoke_pit_log = new G4LogicalVolume(yoke_pit, concrete, "yoke_pit_log");
    auto target_pit_log = new G4LogicalVolume(target_pit, concrete, "target_pit_log");

    // Place the shapes in the world volume
    new G4PVPlacement(new G4ThreeVector(TCC8_shift_x, TCC8_shift_y, TCC8_shift_z), muon_shield_cavern_log, "muon_shield_cavern_phys", logicWorld, false, 0, true);
    new G4PVPlacement(new G4ThreeVector(ECN3_shift_x, ECN3_shift_y, ECN3_shift_z), experiment_cavern_log, "experiment_cavern_phys", logicWorld, false, 0, true);
    new G4PVPlacement(new G4ThreeVector(yoke_pit_shift_x, yoke_pit_shift_y, yoke_pit_shift_z), yoke_pit_log, "yoke_pit_phys", logicWorld, false, 0, true);
    new G4PVPlacement(new G4ThreeVector(target_pit_shift_x, target_pit_shift_y, target_pit_shift_z), target_pit_log, "target_pit_phys", logicWorld, false, 0, true);

    // Create the composite shape
    auto compRock = new G4SubtractionSolid("compRock", rock, muon_shield_cavern, 0, G4ThreeVector(TCC8_shift_x, TCC8_shift_y, TCC8_shift_z));
    compRock = new G4SubtractionSolid("compRock", compRock, experiment_cavern, 0, G4ThreeVector(ECN3_shift_x, ECN3_shift_y, ECN3_shift_z));
    compRock = new G4SubtractionSolid("compRock", compRock, yoke_pit, 0, G4ThreeVector(yoke_pit_shift_x, yoke_pit_shift_y, yoke_pit_shift_z));
    compRock = new G4SubtractionSolid("compRock", compRock, target_pit, 0, G4ThreeVector(target_pit_shift_x, target_pit_shift_y, target_pit_shift_z));

    // Create the logical volume for the composite shape
    auto Cavern = new G4LogicalVolume(compRock, concrete, "Cavern");

    // Place the composite shape in the world volume
    new G4PVPlacement(0, G4ThreeVector(), Cavern, "Cavern_phys", logicWorld, false, 0, true);

    return Cavern;
}*/