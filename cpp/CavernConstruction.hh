#ifndef MY_PROJECT_SHAPECONSTRUCTION_HH
#define MY_PROJECT_SHAPECONSTRUCTION_HH

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SubtractionSolid.hh"
#include "G4ThreeVector.hh"
#include "G4Material.hh"

G4LogicalVolume* ConstructCavern(G4Material* concrete, G4LogicalVolume* logicWorld, double z_transition, double zEndOfAbsorb);

#endif //MY_PROJECT_SHAPECONSTRUCTION_HH