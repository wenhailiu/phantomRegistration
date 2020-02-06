#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iomanip>

#include "itkImage.h"
#include "itkLandmarkBasedTransformInitializer.h"

#include "yaml-cpp/yaml.h"

typedef std::vector< itk::Point<double, 3> > RegPoints;
typedef itk::Point<double, 3> Point;

//Arguments parser:
class InputParser{
    public:
        InputParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }
        /// @author iain
        const std::string& getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }
        /// @author iain
        bool cmdOptionExists(const std::string &option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }
    private:
        std::vector <std::string> tokens;
};

void PrintUsage(std::ostream& os){
    os << "Usage: " << std::endl;
    os << std::setw(15) << "-h:    Print usages. " << std::endl;
    os << std::setw(15) << "-c:    Load configuration file. " << std::endl;
}

int main(int argc, char *argv[]){

    InputParser input(argc, argv);

    if(input.cmdOptionExists("-h") || argc == 1){
        PrintUsage(std::cout);
        return 0;
    }

    std::string fileName;

    if(input.cmdOptionExists("-c")){
        fileName = input.getCmdOption("-c");

        if(fileName.empty()){
            PrintUsage(std::cout);
            return 0;
        }
    }
    else{
        PrintUsage(std::cout);
        return 0;
    }

    YAML::Node parameter_handle = YAML::LoadFile(fileName);
    if(parameter_handle["PhantomLandmarks"].size() != parameter_handle["MocapLandmarks"].size()){
        std::cout << "landmarks do not match!" << std::endl;
        return 0;
    }

    RegPoints fixedITKPoints;
    RegPoints movingITKPoints;
    for(int i = 0; i < parameter_handle["PhantomLandmarks"].size(); ++i){
        auto fixedP = parameter_handle["PhantomLandmarks"]["Landmark#" + std::to_string(i)].as<std::vector<double>>();
        Point fixedITKP(fixedP.data());
        fixedITKPoints.push_back(fixedITKP);

        auto movingP = parameter_handle["MocapLandmarks"]["Landmark#" + std::to_string(i)].as<std::vector<double>>();
        Point movingITKP(movingP.data());
        movingITKPoints.push_back(movingITKP);
    }

    itk::VersorRigid3DTransform<double>::Pointer transform = itk::VersorRigid3DTransform<double>::New();
    transform->SetIdentity();

    itk::LandmarkBasedTransformInitializer< itk::VersorRigid3DTransform<double>, itk::Image<short, 3>, itk::Image<short, 3> >::Pointer initializer = 
    itk::LandmarkBasedTransformInitializer< itk::VersorRigid3DTransform<double>, itk::Image<short, 3>, itk::Image<short, 3> >::New();

    initializer->SetTransform( transform );
    initializer->SetFixedLandmarks( fixedITKPoints );
    initializer->SetMovingLandmarks( movingITKPoints );
    initializer->InitializeTransform();

    std::vector<double> phantomToReferenceTransformMatrix(16, 0.0);

    itk::Matrix<double, 3, 3> transformMatrix = transform->GetMatrix();
    for (unsigned int i = 0; i < transformMatrix.RowDimensions; ++i ){
        for (unsigned int j = 0; j < transformMatrix.ColumnDimensions; ++j ){
            phantomToReferenceTransformMatrix[j + i * 4] = transformMatrix[i][j];
        }
    }

    itk::Vector<double, 3> transformOffset = transform->GetOffset();
    for (unsigned int j = 0; j < transformOffset.GetNumberOfComponents(); ++j ){
        phantomToReferenceTransformMatrix[3 + j * 4] = transformOffset[j];
    }

    phantomToReferenceTransformMatrix[15] = 1.0;

    //Compute error: 
    std::vector<double> Error(fixedITKPoints.size(), 0.0);
    for(int i = 0; i < fixedITKPoints.size(); ++i){
        std::vector<double> currentFixedPoint{fixedITKPoints[i][0], fixedITKPoints[i][1], fixedITKPoints[i][2], 1.0};
        std::vector<double> currentMovingPoint{movingITKPoints[i][0], movingITKPoints[i][1], movingITKPoints[i][2], 1.0};

        std::vector<double> TransformedPoint(3, 0.0);
        for(int row_it = 0; row_it < 3; ++row_it){
            for(int col_it = 0; col_it < 4; ++col_it){
                TransformedPoint[row_it] += phantomToReferenceTransformMatrix[col_it + row_it * 4] * currentFixedPoint[col_it];
            }
        }
        Error[i] = sqrt(
        (TransformedPoint[0] - currentMovingPoint[0]) * (TransformedPoint[0] - currentMovingPoint[0]) + 
        (TransformedPoint[1] - currentMovingPoint[1]) * (TransformedPoint[1] - currentMovingPoint[1]) + 
        (TransformedPoint[2] - currentMovingPoint[2]) * (TransformedPoint[2] - currentMovingPoint[2]));
    }

    double MeanError = 0.0;
    for(int i = 0; i < fixedITKPoints.size(); ++i){
        MeanError += Error[i];
    }
    MeanError /= fixedITKPoints.size();

    parameter_handle["RegistrationResults"]["PhantomToReferenceMatrix"] = phantomToReferenceTransformMatrix;
    parameter_handle["RegistrationResults"]["RegistrationError"] = MeanError;

    std::ofstream OutputHandle(fileName);
    OutputHandle << parameter_handle;
    OutputHandle.close();

    return 0;
}