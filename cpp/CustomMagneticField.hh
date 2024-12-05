#include <vector>
#include "G4ThreeVector.hh"
#include "G4MagneticField.hh"

class CustomMagneticField : public G4MagneticField {
public:
    enum InterpolationType { NEAREST_NEIGHBOR, LINEAR };

    CustomMagneticField(const std::vector<G4ThreeVector>& points, const std::vector<G4ThreeVector>& fields, InterpolationType interpType);
    ~CustomMagneticField();

    void GetFieldValue(const G4double Point[4], G4double *Bfield) const override;
    void GetFieldValueNearestNeighbor(const G4double Point[4], G4double *Bfield) const;
    void GetFieldValueLinear(const G4double Point[4], G4double *Bfield) const;

private:
    std::vector<G4ThreeVector> fPoints;
    std::vector<G4ThreeVector> fFields;
    InterpolationType fInterpType;

    // Grid parameters
    double x_min, y_min, z_min;
    double dx_inv, dy_inv, dz_inv;
    int nx, ny, nz;

    void initializeGrid();
};