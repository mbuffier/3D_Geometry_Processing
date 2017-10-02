#include "acq/evaluation.h"
#include "acq/reconstruction.h"
#include "acq/normalEstimation.h"
#include "acq/ransac.h"

#include "nanogui/formhelper.h"
#include "nanogui/screen.h"

#include "igl/readOBJ.h"
#include "igl/viewer/Viewer.h"

#include <iostream>

namespace acq {

/** \brief                      Re-estimate normals of cloud \p V fitting planes
 *                              to the \p kNeighbours nearest neighbours of each point.
 * \param[in ] kNeighbours      How many neighbours to use (Typiclaly: 5..15)
 * \param[in ] vertices         Input pointcloud. Nx3, where N is the number of points.
 * \param[in ] maxNeighbourDist Maximum distance between vertex and neighbour.
 * \param[out] viewer           The viewer to show the normals at.
 * \return                      The estimated normals, Nx3.
 */
NormalsT
recalcNormals(
    int                 const  kNeighbours,
    CloudT              const& vertices,
    float               const  maxNeighbourDist
) {
    NeighboursT const neighbours =
        calculateCloudNeighbours(
            /* [in]        cloud: */ vertices,
            /* [in] k-neighbours: */ kNeighbours,
            /* [in]      maxDist: */ maxNeighbourDist
        );

    // Estimate normals for points in cloud vertices
    NormalsT normals =
        calculateCloudNormals(
            /* [in]               Cloud: */ vertices,
            /* [in] Lists of neighbours: */ neighbours
        );

    return normals;
} //...recalcNormals()

void setViewerNormals(
    igl::viewer::Viewer      & viewer,
    CloudT              const& vertices,
    NormalsT            const& normals
) {
    // [Optional] Set viewer face normals for shading
    //viewer.data.set_normals(normals);

    // Clear visualized lines (see Viewer.clear())
    viewer.data.lines = Eigen::MatrixXd(0, 9);

    // Add normals to viewer
    viewer.data.add_edges(
        /* [in] Edge starting points: */ vertices,
        /* [in]       Edge endpoints: */ vertices + normals * 0.01, // scale normals to 1% length
        /* [in]               Colors: */ Eigen::Vector3d::Zero()
    );
}

} //...ns acq

int main(int argc, char *argv[]) {

    // How many neighbours to use for normal estimation, shown on GUI.
    int kNeighbours = 10;
    // Maximum distance between vertices to be considered neighbours (FLANN mode)
    float maxNeighbourDist = 0.15; //TODO: set to average vertex distance upon read

    // ********* VARIABLES FOR THE ALGORITHM  ********* 
    int nbIteration = 10000 ; 
    int samplePerIt = 50 ;
    double thresh = 0.001 ;
    double alpha = 0.999 ;
    double thresh_best = 80.0 ;
    float noise = 0.6 ;
    int numberOfOldMesh = 3 ;
    double thresCC = 0.001 ;

    double T_rad = 0.01 ;
    double T_cent = 0.01 ;
    double T_norm = 0.98 ;
    double T_refPt = 0.01 ;

    // will store the current primitives and the point cloud per primitives
    acq::PrimitiveManager best_primitives ;
    acq::CloudManager cloudManagerParts ;

    // deals with several meshes 
    enum MeshType { mesh1=0, mesh2, mesh3} typeMesh = mesh1 ;
    //************************************
    
    // Load a mesh in OFF format
    std::string meshPath1 = "../models/scene_3.off";
    if (argc > 1) {
        meshPath1 = std::string(argv[1]);
        if (meshPath1.find(".obj") == std::string::npos) {
            std::cerr << "Only ready for  OBJ files for now...\n";
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "Usage: iglFrameWork <path-to-off-mesh.obj>." << "\n";
    }

    std::string meshPath2 = "../models/scene_2.off";
    if (argc > 1) {
        meshPath2 = std::string(argv[1]);
        if (meshPath2.find(".obj") == std::string::npos) {
            std::cerr << "Only ready for  OBJ files for now...\n";
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "Usage: iglFrameWork <path-to-off-mesh.obj>." << "\n";
    }

    std::string meshPath3 = "../models/cube_damaged.off";
    if (argc > 1) {
        meshPath2 = std::string(argv[1]);
        if (meshPath2.find(".obj") == std::string::npos) {
            std::cerr << "Only ready for  OBJ files for now...\n";
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "Usage: iglFrameWork <path-to-off-mesh.obj>." << "\n";
    }

    // Visualize the mesh in a viewer
    igl::viewer::Viewer viewer;
    viewer.core.show_lines = false;
    viewer.core.show_overlay = false;

    // Store cloud so we can store normals later
    acq::CloudManager cloudManagerOldMesh;
    // Read mesh from meshPath
    {
        // == ******** For the first mesh ******* ==
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::readOFF(meshPath1, V, F);

        if (V.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath1
                      << "...exiting...\n";
            return EXIT_FAILURE;
        }

        // ----- Normalize Vertices -----
        Eigen::MatrixXd max_row = V.rowwise().maxCoeff();
        Eigen::MatrixXd max_col = V.colwise().maxCoeff();
        V /= std::max(max_row.maxCoeff(), max_col.maxCoeff());

        // == ******** For the second mesh ******* ==

        Eigen::MatrixXd V2;
        Eigen::MatrixXi F2;
        igl::readOFF(meshPath2, V2, F2);

        if (V2.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath2
                      << "...exiting...\n";
            return EXIT_FAILURE;
        }

        // ----- Normalize Vertices -----
        Eigen::MatrixXd max_row2 = V2.rowwise().maxCoeff();
        Eigen::MatrixXd max_col2 = V2.colwise().maxCoeff();
        V2 /= std::max(max_row2.maxCoeff(), max_col2.maxCoeff());

        // == ******** For the third mesh ******* ==

        Eigen::MatrixXd V3;
        Eigen::MatrixXi F3;
        igl::readOFF(meshPath3, V3, F3);

        if (V3.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath3
                      << "...exiting...\n";
            return EXIT_FAILURE;
        }

        // ----- Normalize Vertices -----
        Eigen::MatrixXd max_row3 = V3.rowwise().maxCoeff();
        Eigen::MatrixXd max_col3 = V3.colwise().maxCoeff();
        V3 /= std::max(max_row3.maxCoeff(), max_col3.maxCoeff());

        // 2 times to be able to reload it easily 
        for (int i=0; i<2; i++) {
            cloudManagerOldMesh.addCloud(acq::DecoratedCloud(V, F));
            cloudManagerOldMesh.addCloud(acq::DecoratedCloud(V2, F2));
            cloudManagerOldMesh.addCloud(acq::DecoratedCloud(V3, F3));
        }

        // set the mesh 
        viewer.data.clear() ;

        // Show mesh
        viewer.data.set_mesh(
            cloudManagerOldMesh.getCloud(typeMesh).getVertices(),
            cloudManagerOldMesh.getCloud(typeMesh).getFaces()
        );
    
    }

    // Extend viewer menu using a lambda function
    viewer.callback_init =
        [
            &cloudManagerOldMesh, &kNeighbours, &maxNeighbourDist,
            &nbIteration, &samplePerIt, 
            &best_primitives, &cloudManagerParts, &thresh, &alpha, &thresh_best, 
            &typeMesh, &noise, &numberOfOldMesh, &thresCC, &T_rad, &T_cent, &T_norm, &T_refPt
        ] (igl::viewer::Viewer& viewer)
    {
        // Add an additional menu window
        viewer.ngui->addWindow(Eigen::Vector2i(900,10), "Acquisition3D");

        viewer.ngui->addGroup("Choose your mesh");

        viewer.ngui->addVariable<MeshType>("Which mesh do you want ?",typeMesh)->setItems(
            {"Sphere & Cube","Scene", "Planes"}
        );

        viewer.ngui->addButton("Show the original mesh",
                               [&]() {
            viewer.data.clear() ;
            // Show mesh
            viewer.data.set_mesh(
                             cloudManagerOldMesh.getCloud(typeMesh+numberOfOldMesh).getVertices(),
                             cloudManagerOldMesh.getCloud(typeMesh+numberOfOldMesh).getFaces()
            ); 

            // clean all the primitives   
            best_primitives.clearAllPrimitives() ;
            
            // clear the cloudManager 
            cloudManagerParts.clearCloud() ;

            // replace by the original mesh 
            cloudManagerOldMesh.setCloud(cloudManagerOldMesh.getCloud(typeMesh+numberOfOldMesh), typeMesh) ;
        });        

       viewer.ngui->addButton("Compute Normals",
                               [&]() {

        cloudManagerOldMesh.getCloud(typeMesh+numberOfOldMesh).setNormals(
                acq::recalcNormals(
                        /* [in]      K-neighbours for FLANN: */ kNeighbours,
                        /* [in]             Vertices matrix: */ cloudManagerOldMesh.getCloud(typeMesh+numberOfOldMesh).getVertices(),
                        /* [in]      max neighbour distance: */ maxNeighbourDist
                )
        );

        // Estimate neighbours using FLANN
        acq::NeighboursT const neighbours =
                acq::calculateCloudNeighboursFromFaces(
                        /* [in] Faces: */ cloudManagerOldMesh.getCloud(typeMesh+numberOfOldMesh).getFaces()
                );

        // Estimate normals for points in cloud vertices
        cloudManagerOldMesh.getCloud(typeMesh+numberOfOldMesh).setNormals(
                acq::calculateCloudNormals(
                        /* [in]               Cloud: */ cloudManagerOldMesh.getCloud(typeMesh+numberOfOldMesh).getVertices(),
                        /* [in] Lists of neighbours: */ neighbours
                )
        );

        int nFlips =
                acq::orientCloudNormalsFromFaces(
                        /* [in    ] Lists of neighbours: */ cloudManagerOldMesh.getCloud(typeMesh+numberOfOldMesh).getFaces(),
                        /* [in,out]   Normals to change: */ cloudManagerOldMesh.getCloud(typeMesh+numberOfOldMesh).getNormals()
                );

        // also set to the second mesh 

        // Estimate normals for points in cloud vertices
        cloudManagerOldMesh.getCloud(typeMesh).setNormals(
                            cloudManagerOldMesh.getCloud(typeMesh+numberOfOldMesh).getNormals()
        );


        viewer.data.clear() ;

        // Show mesh
        viewer.data.set_mesh(
            cloudManagerOldMesh.getCloud(typeMesh).getVertices(),
            cloudManagerOldMesh.getCloud(typeMesh).getFaces()
        );

        // Update viewer
        acq::setViewerNormals(
                /* [in, out] Viewer to update: */ viewer,
                /* [in]            Pointcloud: */ cloudManagerOldMesh.getCloud(typeMesh).getVertices(),
                /* [in] Normals of Pointcloud: */ cloudManagerOldMesh.getCloud(typeMesh).getNormals()
        );

        });


        viewer.ngui->addGroup("Choose the parameters");

        // ask for the number of global iteration  
        viewer.ngui->addVariable<int>("Number of iteration : ",[&] (int val) {
                nbIteration = val; }, 
                [&]() { 
                    return nbIteration; 
        } ); 

        viewer.ngui->addVariable<int>("Sample per iteration : ",[&] (int val) {
                samplePerIt = val; }, 
                [&]() { 
                    return samplePerIt; 
        } );  
        
        viewer.ngui->addVariable<double>("Threshold for distance : ",[&] (double val) {
                thresh = val; }, 
                [&]() { 
                    return thresh; 
        } );

        viewer.ngui->addVariable<double>("Threshold for angles : ",[&] (double val) {
                alpha = val; }, 
                [&]() { 
                    return alpha; 
        } );                    
        
        viewer.ngui->addVariable<double>("Threshold for best primitive (%): ",[&] (double val) {
                thresh_best = val; }, 
                [&]() { 
                    return thresh_best; 
        } );   

          viewer.ngui->addButton("RANSAC",
                               [&]() {
            // get back the cloud we want to work on 
            acq::DecoratedCloud thisCloud = cloudManagerOldMesh.getCloud(typeMesh) ;
            int nbVertices = thisCloud.getVertices().rows() ;
            //std::cout << "number of original vertices : " << nbVertices << std::endl ;

             // apply RANSAC 
             bool ransacSuccess = ransac(thisCloud, best_primitives, cloudManagerParts, 
                thresh, alpha, thresh_best, nbIteration, samplePerIt) ;

            if (ransacSuccess) {
                // fuse the result in the new cloud 
                acq::DecoratedCloud* newCloud = gatherClouds(cloudManagerParts,0) ;

                // for evaluation
                /*std::cout << "number of new vertices : " << newCloud->getVertices().rows() << std::endl ;
                float percentage = (float(newCloud->getVertices().rows())/float(nbVertices))*100.f ;
                std::cout << "percentage of detection : " << percentage << std::endl ;*/
                
                viewer.data.clear() ;

                // Show mesh
                viewer.data.set_points(newCloud->getVertices(), newCloud->getColors()) ;
                viewer.core.show_overlay = true;
            }

            else {
                std::cout << "RANSAC didn't find any primitive" << std::endl ;
            }
        });

       viewer.ngui->addVariable<double>("Threshold for radius sphere (%): ",[&] (double val) {
                T_rad = val; }, 
                [&]() { 
                    return T_rad; 
        } );  

        viewer.ngui->addVariable<double>("Threshold for center sphere (%): ",[&] (double val) {
                T_cent = val; }, 
                [&]() { 
                    return T_cent; 
        } );  

        viewer.ngui->addVariable<double>("Threshold normal plane (%): ",[&] (double val) {
                T_norm = val; }, 
                [&]() { 
                    return T_norm; 
        } );  


        viewer.ngui->addVariable<double>("Threshold for reference point (%): ",[&] (double val) {
                T_refPt = val; }, 
                [&]() { 
                    return T_refPt; 
        } );  

          viewer.ngui->addButton("Primitive fusion",
                               [&]() {
            // fuse the similar primitive in cloud manager 
            fuse(best_primitives, cloudManagerParts, T_rad, T_cent, T_norm, T_refPt) ;

           // fuse the result in the new cloud with random color
            acq::DecoratedCloud* newCloud = gatherClouds(cloudManagerParts,0) ;

            // visualisation 
            viewer.data.clear() ;

            // Show mesh
            viewer.data.set_points(newCloud->getVertices(), newCloud->getColors()) ;
            viewer.core.show_overlay = true;

        });     

       viewer.ngui->addVariable<double>("Threshold connective comp :",[&] (double val) {
                thresCC = val; }, 
                [&]() { 
                    return thresCC; 
        } ); 

        viewer.ngui->addButton("Connected components",
                               [&]() {
            connectedComponentManager(cloudManagerParts, best_primitives, thresCC) ;

            // fuse the result in the new cloud with the previous color computed in connected comp
            acq::DecoratedCloud* newCloud = gatherClouds(cloudManagerParts,1);

            viewer.data.clear() ;

            // Show mesh
            viewer.data.set_points(newCloud->getVertices(), newCloud->getColors()) ;
            viewer.core.show_overlay = true;
        });

        /// ----- RECONSTRUCTION ----
        viewer.ngui->addButton("Reconstruction", [&]() {

            int nbSample = 1000;
            double T = 0.1;

            acq::DecoratedCloud* newCloud = gatherClouds(cloudManagerParts, 0) ;
            acq::DecoratedCloud cloud = acq::DecoratedCloud(newCloud->getVertices(),newCloud->getNormals(),newCloud->getColors());

            reconstruct(best_primitives, cloud, nbSample, thresh, alpha, T);

            // Show mesh
            viewer.data.clear() ;
            viewer.data.set_points(cloud.getVertices(), cloud.getColors()) ;
            viewer.core.show_overlay = true;
            });

        viewer.ngui->addGroup("Test noise");
       
       viewer.ngui->addVariable<float>( "% of noise", [&] (float val) {
                noise = val;}, [&]() {
                return noise; // get
       } );

       // adding noise for evaluation 
       viewer.ngui->addButton("Add noise",[&]() {
        // noise the position of the vertex 
        cloudManagerOldMesh.getCloud(typeMesh).setVertices(addNoise(noise,cloudManagerOldMesh.getCloud(typeMesh),1)) ;

        // noise the normals 
        cloudManagerOldMesh.getCloud(typeMesh).setNormals(addNoise(noise,cloudManagerOldMesh.getCloud(typeMesh),2));  

        viewer.data.clear() ;

        viewer.data.set_mesh(
            cloudManagerOldMesh.getCloud(typeMesh).getVertices(),
            cloudManagerOldMesh.getCloud(typeMesh).getFaces()) ;
        }) ;

        // Generate menu
        viewer.screen->performLayout();

        return false;
    }; //...viewer menu


    // Start viewer
    viewer.launch();

    return 0;
} //...main()
