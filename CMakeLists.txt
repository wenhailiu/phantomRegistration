project(phantomRegistration)

#Setting for VTK:
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

#Setting for ITK:
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

#Setting yaml-cpp:
find_package(yaml-cpp REQUIRED)

add_executable(
    PhantomRegistrationTool 
    phantomRegistration.cpp
)

target_link_libraries(
    PhantomRegistrationTool 
    ${VTK_LIBRARIES} 
    ${ITK_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
)
