#ifndef CUSTOM_MAGNETIC_FIELD_HH
#define CUSTOM_MAGNETIC_FIELD_HH

#include "G4MagneticField.hh"
#include "G4ThreeVector.hh"
#include <vector>

class CustomMagneticField : public G4MagneticField {
public:
    enum InterpolationType {
        NEAREST_NEIGHBOR,
        LINEAR
    };

    CustomMagneticField(const std::vector<G4ThreeVector>& points, const std::vector<G4ThreeVector>& fields, InterpolationType interpType);
    virtual ~CustomMagneticField();

    virtual void GetFieldValue(const G4double Point[4], G4double *Bfield) const override;

private:
    std::vector<G4ThreeVector> fPoints;
    std::vector<G4ThreeVector> fFields;
    InterpolationType fInterpType;

    void GetFieldValueNearestNeighbor(const G4double Point[4], G4double *Bfield) const;
    void GetFieldValueLinear(const G4double Point[4], G4double *Bfield) const;
};

#endif // CUSTOM_MAGNETIC_FIELD_HH