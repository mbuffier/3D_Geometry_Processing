#include "acq/normalEstimation.h"
#include "acq/cloudManager.h"
#include "acq/decoratedCloud.h"
#include "acq/discreteCurvature.h"
#include "acq/smoothing.h"

#include "nanogui/formhelper.h"
#include "nanogui/screen.h"

#include "igl/readOFF.h"
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
    double pourcentage = 0.0 ;
    double lambda = 0.0 ;
    bool typeDiscretization = false ;
    float noise ;
    // Dummy enum to demo GUI
    enum MeshType { Bumpty=0, Bunny, Fandisk, Cow, Dragon} typeMesh = Bumpty ;

    // Dummy variable to demo GUI
    bool boolVariable = true;
    // Dummy variable to demo GUI
    float floatVariable = 0.1f;

    // Load a mesh in OFF format
    std::string meshPath = "../3rdparty/libigl/tutorial/shared/screwdriver.off";
    if (argc > 1) {
        meshPath = std::string(argv[1]);
        if (meshPath.find(".off") == std::string::npos) {
            std::cerr << "Only ready for  OFF files for now...\n";
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "Usage: iglFrameWork <path-to-off-mesh.off>." << "\n";
    }

        std::string meshPath2 = "../bunny.off";
    if (argc > 1) {
        meshPath2 = std::string(argv[1]);
        if (meshPath2.find(".off") == std::string::npos) {
            std::cerr << "Only ready for  OFF files for now...\n";
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "Usage: iglFrameWork <path-to-off-mesh.off>." << "\n";
    }

        std::string meshPath3 = "../bumpy.off";
    if (argc > 1) {
        meshPath3 = std::string(argv[1]);
        if (meshPath3.find(".off") == std::string::npos) {
            std::cerr << "Only ready for  OFF files for now...\n";
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "Usage: iglFrameWork <path-to-off-mesh.off>." << "\n";
    }
    
    std::string meshPath4 = "../3rdparty/libigl/tutorial/shared/cow.off";
    if (argc > 1) {
        meshPath3 = std::string(argv[1]);
        if (meshPath4.find(".off") == std::string::npos) {
            std::cerr << "Only ready for  OFF files for now...\n";
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "Usage: iglFrameWork <path-to-off-mesh.off>." << "\n";
    }

    std::string meshPath5 = "../dragon.off";
    if (argc > 1) {
        meshPath3 = std::string(argv[1]);
        if (meshPath5.find(".off") == std::string::npos) {
            std::cerr << "Only ready for  OFF files for now...\n";
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "Usage: iglFrameWork <path-to-off-mesh.off>." << "\n";
    }

    // Visualize the mesh in a viewer
    igl::viewer::Viewer viewer;
    {
        // Don't show face edges
        viewer.core.show_lines = false;
        viewer.core.show_overlay = false;
        viewer.core.invert_normals = false;
    }

    // Store cloud so we can store normals later
    acq::CloudManager cloudManager;
    // Read mesh from meshPath
    {
        // Pointcloud vertices, N rows x 3 columns.
        Eigen::MatrixXd V;
        // Face indices, M x 3 integers referring to V.
        Eigen::MatrixXi F;
        // Read mesh
        igl::readOFF(meshPath, V, F);
        // Check, if any vertices read
        if (V.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath
            << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read

        // Pointcloud vertices, N rows x 3 columns.
        Eigen::MatrixXd V2;
        // Face indices, M x 3 integers referring to V.
        Eigen::MatrixXi F2;
        // Read mesh
        igl::readOFF(meshPath2, V2, F2);
        // Check, if any vertices read
        if (V2.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath2
            << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read

                // Pointcloud vertices, N rows x 3 columns.
        Eigen::MatrixXd V3;
        // Face indices, M x 3 integers referring to V.
        Eigen::MatrixXi F3;
        // Read mesh
        igl::readOFF(meshPath3, V3, F3);
        // Check, if any vertices read
        if (V3.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath3
            << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read

        // Pointcloud vertices, N rows x 3 columns.
        Eigen::MatrixXd V4;
        // Face indices, M x 3 integers referring to V.
        Eigen::MatrixXi F4;
        // Read mesh
        igl::readOFF(meshPath4, V4, F4);
        // Check, if any vertices read
        if (V4.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath3
            << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read

        Eigen::MatrixXd V5;
        // Face indices, M x 3 integers referring to V.
        Eigen::MatrixXi F5;
        // Read mesh
        igl::readOFF(meshPath5, V5, F5);
        // Check, if any vertices read
        if (V5.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath5
            << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read

        for (int i=0; i<2; i++) {
            cloudManager.addCloud(acq::DecoratedCloud(V3, F3));  
            cloudManager.addCloud(acq::DecoratedCloud(V2, F2));  
            cloudManager.addCloud(acq::DecoratedCloud(V, F));   
            cloudManager.addCloud(acq::DecoratedCloud(V4, F4));     
            cloudManager.addCloud(acq::DecoratedCloud(V5, F5));                     
        }
        // Show mesh
        viewer.data.set_mesh(
                             cloudManager.getCloud(typeMesh).getVertices(),
                             cloudManager.getCloud(typeMesh).getFaces()
                             );


    } //...read mesh
    
    // Extend viewer menu using a lambda function
    viewer.callback_init =
    [
     &cloudManager, &kNeighbours, &maxNeighbourDist, &noise,
     &floatVariable, &boolVariable, &typeMesh, &lambda, &pourcentage, &typeDiscretization
     ] (igl::viewer::Viewer& viewer)
    {
        // Add an additional menu window
        viewer.ngui->addWindow(Eigen::Vector2i(900,10), "Acquisition3D");
        
        viewer.ngui->addGroup("Choose your mesh");

        viewer.ngui->addVariable<MeshType>("Which mesh do you want ?",typeMesh)->setItems(
            {"Bumpty","Bunny","Screwdriver", "Cow", "Dragon"}
        );

        viewer.ngui->addButton("Show new mesh",
                               [&]() {
        viewer.data.clear() ;

            // Show mesh
        viewer.data.set_mesh(
                             cloudManager.getCloud(typeMesh).getVertices(),
                             cloudManager.getCloud(typeMesh).getFaces()
                             );                                 
        });

       viewer.ngui->addButton("Back to the original mesh",
                               [&]() {
        cloudManager.setCloud(cloudManager.getCloud(typeMesh+5), typeMesh) ;                
        viewer.data.clear() ;

            // Show mesh
        viewer.data.set_mesh(
                             cloudManager.getCloud(typeMesh).getVertices(),
                             cloudManager.getCloud(typeMesh).getFaces()
                             );                                 
        });

        viewer.ngui->addGroup("Set normals for the mesh");

        viewer.ngui->addButton("Estimate normals (from faces)",[&]() {

                if (!cloudManager.getCloud(typeMesh).hasNormals())
                    cloudManager.getCloud(typeMesh).setNormals(
                        acq::recalcNormals(
                            /* [in]      K-neighbours for FLANN: */ kNeighbours,
                            /* [in]             Vertices matrix: */ cloudManager.getCloud(typeMesh).getVertices(),
                            /* [in]      max neighbour distance: */ maxNeighbourDist
                        )
                    );

                // Estimate neighbours using FLANN
                acq::NeighboursT const neighbours =
                    acq::calculateCloudNeighboursFromFaces(
                        /* [in] Faces: */ cloudManager.getCloud(typeMesh).getFaces()
                    );

                // Estimate normals for points in cloud vertices
                cloudManager.getCloud(typeMesh).setNormals(
                    acq::calculateCloudNormals(
                        /* [in]               Cloud: */ cloudManager.getCloud(typeMesh).getVertices(),
                        /* [in] Lists of neighbours: */ neighbours
                    )
                );

                // Update viewer
                acq::setViewerNormals(
                    /* [in, out] Viewer to update: */ viewer,
                    /* [in]            Pointcloud: */ cloudManager.getCloud(typeMesh).getVertices(),
                    /* [in] Normals of Pointcloud: */ cloudManager.getCloud(typeMesh).getNormals()
                );
            } //...button push lambda
        ); //...estimate normals from faces

        // Add a button for orienting normals using face information
        viewer.ngui->addButton(
            /* Displayed label: */ "Orient normals (from faces)",

            /* Lambda to call: */ [&]() {
                // Check, if normals already exist
                if (!cloudManager.getCloud(typeMesh).hasNormals())
                    cloudManager.getCloud(typeMesh).setNormals(
                        acq::recalcNormals(
                            /* [in]      K-neighbours for FLANN: */ kNeighbours,
                            /* [in]             Vertices matrix: */ cloudManager.getCloud(typeMesh).getVertices(),
                            /* [in]      max neighbour distance: */ maxNeighbourDist
                        )
                    );

                // Orient normals in place using established neighbourhood
                int nFlips =
                    acq::orientCloudNormalsFromFaces(
                        /* [in    ] Lists of neighbours: */ cloudManager.getCloud(typeMesh).getFaces(),
                        /* [in,out]   Normals to change: */ cloudManager.getCloud(typeMesh).getNormals()
                    );

                // Update viewer
                acq::setViewerNormals(
                    /* [in, out] Viewer to update: */ viewer,
                    /* [in]            Pointcloud: */ cloudManager.getCloud(typeMesh).getVertices(),
                    /* [in] Normals of Pointcloud: */ cloudManager.getCloud(typeMesh).getNormals()
                );
            } 
); 

    
        // Add new group
        viewer.ngui->addGroup("Mean curvature");
        
        viewer.ngui->addButton("Using uniform discretization",
                               [&]() {
                                   // apply the mean curvature uniform
        acq::DecoratedCloud &cloud = cloudManager.getCloud(typeMesh);
        Eigen::MatrixXd color ; 
        color = meanCurvature(cloud, 0) ;

        // change the color of the mesh
        viewer.data.set_colors(color);
                                   
        });
        
        viewer.ngui->addButton("Using cotan discretization",
                               [&]() {
        acq::DecoratedCloud &cloud = cloudManager.getCloud(typeMesh);
        Eigen::MatrixXd color ; 
        color = meanCurvature(cloud, 1) ;

        viewer.data.set_colors(color);                                   
        });


        viewer.ngui->addGroup("Gaussian curvature");

        viewer.ngui->addButton("Discrete Gaussian curvature",
                               [&]() {
             acq::DecoratedCloud &cloud = cloudManager.getCloud(typeMesh);
             Eigen::MatrixXd color ; 
             color = gaussianCurvUnifom(cloud) ;
             viewer.data.set_colors(color);
        });                                   

        viewer.ngui->addGroup("Smoothing");

        // ask for the poucentage of the bounding box instead of lambda 
        viewer.ngui->addVariable<float>("% of the diagonal BB (int) : ",[&] (double val) {
                pourcentage = val; 
                lambda = findLambda(pourcentage,cloudManager.getCloud(typeMesh)) ; }, 
                [&]() { 
                    return lambda; 
                } ); 


         viewer.ngui->addVariable<bool>(
            "Cotan distretization : ",
            [&](bool val) {
                typeDiscretization = val; // set
            },
            [&]() {
                return typeDiscretization; // get
            });


          viewer.ngui->addButton("Explicit smoothing",
                               [&]() {

            // set the vertices 
             cloudManager.getCloud(typeMesh).setVertices(explicitSmoothing(cloudManager.getCloud(typeMesh), lambda,typeDiscretization)) ;

             // set the normals 
             acq::NeighboursT const neighbours =
                    acq::calculateCloudNeighboursFromFaces(
                       cloudManager.getCloud(typeMesh).getFaces()
                    );

                // Estimate normals for points in cloud vertices
                cloudManager.getCloud(typeMesh).setNormals(
                    acq::calculateCloudNormals(
                        cloudManager.getCloud(typeMesh).getVertices(),
                        neighbours
                    )
                );

                int nFlips =
                    acq::orientCloudNormalsFromFaces(
                         cloudManager.getCloud(typeMesh).getFaces(),
                         cloudManager.getCloud(typeMesh).getNormals()
                    );

             viewer.data.clear() ;

            // Show mesh
            viewer.data.set_mesh(
                             cloudManager.getCloud(typeMesh).getVertices(),
                             cloudManager.getCloud(typeMesh).getFaces()
                             );

             Eigen::MatrixXd color ; 
             color = meanCurvature(cloudManager.getCloud(typeMesh), 1) ;

             viewer.data.set_colors(color) ;
 
        });   

          viewer.ngui->addButton("Implicit smoothing",
                               [&]() {
            // set the vertices 
             cloudManager.getCloud(typeMesh).setVertices(implicitSmoothing(cloudManager.getCloud(typeMesh), lambda)) ;

             // set the normals 
             acq::NeighboursT const neighbours =
                    acq::calculateCloudNeighboursFromFaces(
                       cloudManager.getCloud(typeMesh).getFaces()
                    );


                cloudManager.getCloud(typeMesh).setNormals(
                    acq::calculateCloudNormals(
                        cloudManager.getCloud(typeMesh).getVertices(),
                        neighbours
                    )
                );

                int nFlips =
                    acq::orientCloudNormalsFromFaces(
                         cloudManager.getCloud(typeMesh).getFaces(),
                         cloudManager.getCloud(typeMesh).getNormals()
                    );

             viewer.data.clear() ;

            // Show mesh
            viewer.data.set_mesh(
                             cloudManager.getCloud(typeMesh).getVertices(),
                             cloudManager.getCloud(typeMesh).getFaces()
                             );

             Eigen::MatrixXd color ; 
             color = meanCurvature(cloudManager.getCloud(typeMesh), 1) ;
             viewer.data.set_colors(color);

        });
        
                viewer.ngui->addGroup("Test denoising");


       viewer.ngui->addVariable<float>( "% of noise", [&] (float val) {
                noise = val;}, [&]() {
                return noise; // get
       } );

       viewer.ngui->addButton("Add noise",[&]() {
        cloudManager.getCloud(typeMesh).setVertices(addNoise(noise, cloudManager.getCloud(typeMesh))) ;

        cloudManager.getCloud(typeMesh).setNormals(
            acq::recalcNormals(
                 kNeighbours,
                 cloudManager.getCloud(typeMesh).getVertices(),
                 maxNeighbourDist
            )
        );
        // Show mesh
        viewer.data.clear() ;
        viewer.data.set_mesh(
            cloudManager.getCloud(typeMesh).getVertices(),
            cloudManager.getCloud(typeMesh).getFaces()
        );
        viewer.data.set_colors(cloudManager.getCloud(typeMesh).getColors());        
        });


        viewer.ngui->addButton("Compute the error",[&]() {
            float error ; 
            error = computeError(cloudManager.getCloud(typeMesh),cloudManager.getCloud(typeMesh+5)) ;
            std::cout << "The error is : " << error << std::endl ;
        });
                                   
         // Generate menu
         viewer.screen->performLayout();                         
         return false;
          }; //...viewer menu
                               
         // Start viewer
         viewer.launch();
         return 0;
        } //...main()
