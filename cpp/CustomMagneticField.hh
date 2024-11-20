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

    // Binning structure
    struct Bin {
        G4ThreeVector minCorner;
        G4ThreeVector maxCorner;
        std::vector<size_t> nearestNeighbors;
    };

    std::vector<Bin> bins;
    size_t numBinsX, numBinsY, numBinsZ;
    G4ThreeVector binSize;

    void initializeBins();
    void findBin(const G4ThreeVector& point, size_t& binX, size_t& binY, size_t& binZ) const;
};