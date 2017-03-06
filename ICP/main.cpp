#include "acq/normalEstimation.h"
#include "acq/decoratedCloud.h"
#include "acq/cloudManager.h"
#include <ANN/ANN.h>					// ANN declarations

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

// function to transform a matrix into a ANNpointArray
ANNpointArray matrixToANNArray(Eigen::MatrixXd const& points) {
    unsigned int M = points.rows() ;
    int dim = 3 ;

	ANNpointArray dataPts;	

    dataPts = annAllocPts(M, dim);			// allocate data points

    for (int i=0; i<M; i++) {
        ANNpoint point ;
        point = annAllocPt(dim) ;

        for (int j=0; j<dim; j++) {
            point[j] = points(i,j) ;
        }
        dataPts[i] = point ;
    }
    return dataPts ;
}


int main(int argc, char *argv[]) {

    // How many neighbours to use for normal estimation, shown on GUI.
    int kNeighbours = 10;
    // Maximum distance between vertices to be considered neighbours (FLANN mode)
    float maxNeighbourDist = 0.15; //TODO: set to average vertex distance upon read

    // to rotate the mesh 
    float angle = 0 ;

    // Dummy enum to demo GUI
    enum Orientation { Up=0, Down, Left, Right } dir = Up;
    // Dummy variable to demo GUI
    bool boolVariable = true;
    // Dummy variable to demo GUI
    float floatVariable = 0.1f;

    // Load a mesh in OFF format
    std::string meshPath = "../images/bun000.off";
    if (argc > 1) {
        meshPath = std::string(argv[1]);
        if (meshPath.find(".off") == std::string::npos) {
            std::cerr << "Only ready for  OFF files for now...\n";
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "Usage: iglFrameWork <path-to-off-mesh.off>." << "\n";
    }

    std::string meshPath2 = "../images/bun045.off";
    if (argc > 1) {
        meshPath2 = std::string(argv[1]);
        if (meshPath2.find(".off") == std::string::npos) {
            std::cerr << "Only ready for  OFF files for now...\n";
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "Usage: iglFrameWork <path-to-off-mesh.off>." << "\n";
    }

    // load multiple bunnies 
    std::string meshPath3 = "../images/bun090.off";
    if (argc > 1) {
        meshPath3 = std::string(argv[1]);
        if (meshPath3.find(".off") == std::string::npos) {
            std::cerr << "Only ready for  OFF files for now...\n";
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "Usage: iglFrameWork <path-to-off-mesh.off>." << "\n";
    }

    std::string meshPath4 = "../images/bun180.off";
    if (argc > 1) {
        meshPath4 = std::string(argv[1]);
        if (meshPath4.find(".off") == std::string::npos) {
            std::cerr << "Only ready for  OFF files for now...\n";
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "Usage: iglFrameWork <path-to-off-mesh.off>." << "\n";
    }

    std::string meshPath5 = "../images/bun270.off";
    if (argc > 1) {
        meshPath5 = std::string(argv[1]);
        if (meshPath5.find(".off") == std::string::npos) {
            std::cerr << "Only ready for  OFF files for now...\n";
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "Usage: iglFrameWork <path-to-off-mesh.off>." << "\n";
    }

    std::string meshPath6 = "../images/bun315.off";
    if (argc > 1) {
        meshPath6 = std::string(argv[1]);
        if (meshPath6.find(".off") == std::string::npos) {
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
    }

    // Store cloud so we can store normals later
    acq::CloudManager cloudManager ;
    ANNkd_tree*	kdTreeConst ;
    int const nbP = 6000 ; //nbp total : 40097

    // to add the noise after 
    float sigmaX   ;
    float sigmaY  ;
    float sigmaZ ;

    // Read mesh from meshPath
    {
        // Pointcloud vertices, N rows x 3 columns.
        Eigen::MatrixXd V;
        Eigen::MatrixXd V2;
        Eigen::MatrixXd V3;
        Eigen::MatrixXd V4;
        Eigen::MatrixXd V5;
        Eigen::MatrixXd V6;

        // Face indices, M x 3 integers referring to V.
        Eigen::MatrixXi F;
        Eigen::MatrixXi F2;
        Eigen::MatrixXi F3;
        Eigen::MatrixXi F4;
        Eigen::MatrixXi F5;
        Eigen::MatrixXi F6;

        // Read all the meshes 
        igl::readOFF(meshPath, V, F);
        // Check, if any vertices read
        if (V.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath
                      << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read

        igl::readOFF(meshPath2, V2, F2);
        // Check, if any vertices read
        if (V2.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath
                      << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read

        igl::readOFF(meshPath3, V3, F3);
        // Check, if any vertices read
        if (V3.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath
                      << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read

        igl::readOFF(meshPath4, V4, F4);
        // Check, if any vertices read
        if (V4.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath
                      << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read

        igl::readOFF(meshPath5, V5, F5);
        // Check, if any vertices read
        if (V5.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath
                      << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read

        igl::readOFF(meshPath6, V6, F6);
        // Check, if any vertices read
        if (V6.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath
                      << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read

        // Store read vertices and faces
        cloudManager.addCloud(acq::DecoratedCloud(V, F));
        cloudManager.addCloud(acq::DecoratedCloud(V2, F2));
        cloudManager.addCloud(acq::DecoratedCloud(V2, F2));

        // print informations about the number of points used for the ICP algorithm
        // print the result for this search 
         float pour = (float(nbP)/float(cloudManager.getCloud(1).getVertices().rows()))*100.0 ;
        std::cout<< "Number of points sampled :  " << nbP << " over " <<  cloudManager.getCloud(1).getVertices().rows() << " points in the mesh (" << pour <<"%)"<< std::endl ;

        // define the kd tree construction 
        int dim = cloudManager.getCloud(0).getVertices().cols() ;
        int nPts = cloudManager.getCloud(0).getVertices().rows() ;

        // construct the array of fixed point to built the kdTree
        ANNpointArray vertices2Match ;  
        vertices2Match = annAllocPts(nPts, dim);
        vertices2Match = matrixToANNArray(cloudManager.getCloud(0).getVertices()) ; 
        
         kdTreeConst = new ANNkd_tree(					// build search structure
					vertices2Match,					// the data points
					nPts,						// number of points
					dim);						// dimension of space

        // compute the value of the boundingBox for the second cloud 
        float Xmax, Xmin, Ymax, Ymin, Zmax, Zmin ;
        cloudManager.getCloud(1).boundingBox(Xmax, Xmin, Ymax, Ymin,Zmax, Zmin) ;

        // set the variance to sigma = 0.08% of the bouding box size in each direction
        sigmaX =(Xmax-Xmin)*0.0008  ;
        sigmaY = (Ymax-Ymin)*0.0008 ;
        sigmaZ = (Zmax-Zmin)*0.0008 ;

        // Calculate normals on launch for the 3 meshes
        cloudManager.getCloud(0).setNormals(
            acq::recalcNormals(
                /* [in]      K-neighbours for FLANN: */ kNeighbours,
                /* [in]             Vertices matrix: */ cloudManager.getCloud(0).getVertices(),
                /* [in]      max neighbour distance: */ maxNeighbourDist
            )
        );

        cloudManager.getCloud(1).setNormals(
            acq::recalcNormals(
                /* [in]      K-neighbours for FLANN: */ kNeighbours,
                /* [in]             Vertices matrix: */ cloudManager.getCloud(1).getVertices(),
                /* [in]      max neighbour distance: */ maxNeighbourDist
            )
        );
        
        cloudManager.getCloud(2).setNormals(
            acq::recalcNormals(
                /* [in]      K-neighbours for FLANN: */ kNeighbours,
                /* [in]             Vertices matrix: */ cloudManager.getCloud(2).getVertices(),
                /* [in]      max neighbour distance: */ maxNeighbourDist
            )
        );

        // add the mesh combined 
        cloudManager.addCloud(acq::DecoratedCloud(cloudManager.getCloud(0), cloudManager.getCloud(1), false)) ;

        // add meshes to complete the bunny 
        cloudManager.addCloud(acq::DecoratedCloud(V3, F3));
        cloudManager.addCloud(acq::DecoratedCloud(V4, F4));
        cloudManager.addCloud(acq::DecoratedCloud(V5, F5));
        cloudManager.addCloud(acq::DecoratedCloud(V6, F6));

        // compute their normals
        cloudManager.getCloud(4).setNormals(
            acq::recalcNormals(
                /* [in]      K-neighbours for FLANN: */ kNeighbours,
                /* [in]             Vertices matrix: */ cloudManager.getCloud(4).getVertices(),
                /* [in]      max neighbour distance: */ maxNeighbourDist
            )
        );

        cloudManager.getCloud(5).setNormals(
            acq::recalcNormals(
                /* [in]      K-neighbours for FLANN: */ kNeighbours,
                /* [in]             Vertices matrix: */ cloudManager.getCloud(5).getVertices(),
                /* [in]      max neighbour distance: */ maxNeighbourDist
            )
        );

        cloudManager.getCloud(6).setNormals(
            acq::recalcNormals(
                /* [in]      K-neighbours for FLANN: */ kNeighbours,
                /* [in]             Vertices matrix: */ cloudManager.getCloud(6).getVertices(),
                /* [in]      max neighbour distance: */ maxNeighbourDist
            )
        );

        cloudManager.getCloud(7).setNormals(
            acq::recalcNormals(
                /* [in]      K-neighbours for FLANN: */ kNeighbours,
                /* [in]             Vertices matrix: */ cloudManager.getCloud(6).getVertices(),
                /* [in]      max neighbour distance: */ maxNeighbourDist
            )
        );

        // Show mesh
        viewer.data.set_mesh(
            cloudManager.getCloud(3).getVertices(),
            cloudManager.getCloud(3).getFaces()
        );

        viewer.data.set_colors(cloudManager.getCloud(3).getColors());

        // Update viewer
        acq::setViewerNormals(
            viewer,
            cloudManager.getCloud(3).getVertices(),
            cloudManager.getCloud(3).getNormals()
        );

    } //...read mesh

    // Extend viewer menu using a lambda function
    viewer.callback_init =
        [
            &cloudManager, &kNeighbours, &maxNeighbourDist,
            &floatVariable, &boolVariable, &dir, &kdTreeConst, &nbP, &angle, 
            &sigmaX, &sigmaY, &sigmaZ 
        ] (igl::viewer::Viewer& viewer)
    {
        // Add an additional menu window
        viewer.ngui->addWindow(Eigen::Vector2i(900,10), "Acquisition3D");

        // Add new group
        viewer.ngui->addGroup("Old functions");

        // Add k-neighbours variable to GUI
        viewer.ngui->addVariable<int>(
            /* Displayed name: */ "k-neighbours",

            /*  Setter lambda: */ [&] (int val) {
                // Store reference to current cloud (id 0 for now)

                // Store new value
                kNeighbours = val;

                // Recalculate normals for cloud and update viewer
                cloudManager.getCloud(0).setNormals(
                    acq::recalcNormals(
                        kNeighbours,
                        cloudManager.getCloud(0).getVertices(),
                       maxNeighbourDist
                    )
                );

                // Recalculate normals for cloud and update viewer
                cloudManager.getCloud(2).setNormals(
                    acq::recalcNormals(
                        /* [in]      K-neighbours for FLANN: */ kNeighbours,
                        /* [in]             Vertices matrix: */ cloudManager.getCloud(2).getVertices(),
                        /* [in]      max neighbour distance: */ maxNeighbourDist
                    )
                );

                 cloudManager.setCloud(acq::DecoratedCloud(cloudManager.getCloud(0), cloudManager.getCloud(2), false),3) ;


                // Update viewer with the combined mesh 
                acq::setViewerNormals(
                    /* [in, out] Viewer to update: */ viewer,
                    /* [in]            Pointcloud: */ cloudManager.getCloud(3).getVertices(),
                    /* [in] Normals of Pointcloud: */ cloudManager.getCloud(3).getNormals()
                );
            }, //...setter lambda

            /*  Getter lambda: */ [&]() {
                return kNeighbours; // get
            } //...getter lambda
        ); //...addVariable(kNeighbours)

        // Add maxNeighbourDistance variable to GUI
        viewer.ngui->addVariable<float>(
            /* Displayed name: */ "maxNeighDist",

            /*  Setter lambda: */ [&] (float val) {
                // Store new value
                maxNeighbourDist = val;

                // Recalculate normals for cloud and update viewer
                cloudManager.getCloud(0).setNormals(
                    acq::recalcNormals(
                        /* [in]      K-neighbours for FLANN: */ kNeighbours,
                        /* [in]             Vertices matrix: */ cloudManager.getCloud(0).getVertices(),
                        /* [in]      max neighbour distance: */ maxNeighbourDist
                    )
                );

                cloudManager.getCloud(2).setNormals(
                    acq::recalcNormals(
                        /* [in]      K-neighbours for FLANN: */ kNeighbours,
                        /* [in]             Vertices matrix: */ cloudManager.getCloud(2).getVertices(),
                        /* [in]      max neighbour distance: */ maxNeighbourDist
                    )
                );

                 cloudManager.setCloud(acq::DecoratedCloud(cloudManager.getCloud(0), cloudManager.getCloud(2), false),3) ;

                // Update viewer
                acq::setViewerNormals(
                    /* [in, out] Viewer to update: */ viewer,
                    /* [in]            Pointcloud: */ cloudManager.getCloud(3).getVertices(),
                    /* [in] Normals of Pointcloud: */ cloudManager.getCloud(3).getNormals()
                );
            }, //...setter lambda

            /*  Getter lambda: */ [&]() {
                return maxNeighbourDist; // get
            } //...getter lambda
        ); //...addVariable(kNeighbours)

        // Add a button for estimating normals using FLANN as neighbourhood
        // same, as changing kNeighbours
        viewer.ngui->addButton(
            /* displayed label: */ "Estimate normals (FLANN)",

            /* lambda to call: */ [&]() {

                acq::DecoratedCloud & cloud0 = cloudManager.getCloud(0 ) ;
                acq::DecoratedCloud & cloud1 = cloudManager.getCloud(2 ) ;

                // Recalculate normals for cloud and update viewer
                cloud0.setNormals(
                    acq::recalcNormals(
                        /* [in]      K-neighbours for FLANN: */ kNeighbours,
                        /* [in]             Vertices matrix: */ cloud0.getVertices(),
                        /* [in]      max neighbour distance: */ maxNeighbourDist
                    )
                );

                cloud1.setNormals(
                    acq::recalcNormals(
                        /* [in]      k-neighbours for flann: */ kNeighbours,
                        /* [in]             vertices matrix: */ cloud1.getVertices(),
                        /* [in]      max neighbour distance: */ maxNeighbourDist
                    )
                );

                 cloudManager.setCloud(acq::DecoratedCloud(cloud0,cloud1, false),3) ;

                // update viewer
                acq::setViewerNormals(
                    /* [in, out] viewer to update: */ viewer,
                    /* [in]            pointcloud: */ cloudManager.getCloud(3).getVertices(),
                    /* [in] normals of pointcloud: */ cloudManager.getCloud(3).getNormals()
                );
            } //...button push lambda
        ); //...estimate normals using FLANN

        // Add a button for orienting normals using FLANN
        viewer.ngui->addButton(
            /* Displayed label: */ "Orient normals (FLANN)",

            /* Lambda to call: */ [&]() {

                // for the first mesh 
                if (!cloudManager.getCloud(0).hasNormals())
                    cloudManager.getCloud(0).setNormals(
                        acq::recalcNormals(
                            /* [in]      K-neighbours for FLANN: */ kNeighbours,
                            /* [in]             Vertices matrix: */ cloudManager.getCloud(0).getVertices(),
                            /* [in]      max neighbour distance: */ maxNeighbourDist
                        )
                    );

                // Estimate neighbours using FLANN
                acq::NeighboursT const neighbours =
                    acq::calculateCloudNeighbours(
                        /* [in]        Cloud: */ cloudManager.getCloud(0).getVertices(),
                        /* [in] k-neighbours: */ kNeighbours,
                        /* [in]      maxDist: */ maxNeighbourDist
                    );

                // Orient normals in place using established neighbourhood
                int nFlips =
                    acq::orientCloudNormals(
                        /* [in    ] Lists of neighbours: */ neighbours,
                        /* [in,out]   Normals to change: */ cloudManager.getCloud(0).getNormals()
                    );
                std::cout << "nFlips: " << nFlips << "/" <<  cloudManager.getCloud(0).getNormals().size() << "\n";

                // for the second mesh 
                if (!cloudManager.getCloud(2).hasNormals())
                    cloudManager.getCloud(2).setNormals(
                        acq::recalcNormals(
                            /* [in]      K-neighbours for FLANN: */ kNeighbours,
                            /* [in]             Vertices matrix: */ cloudManager.getCloud(2).getVertices(),
                            /* [in]      max neighbour distance: */ maxNeighbourDist
                        )
                    );

                // Estimate neighbours using FLANN
                acq::NeighboursT const neighbours2 =
                    acq::calculateCloudNeighbours(
                        /* [in]        Cloud: */ cloudManager.getCloud(2).getVertices(),
                        /* [in] k-neighbours: */ kNeighbours,
                        /* [in]      maxDist: */ maxNeighbourDist
                    );

                // Orient normals in place using established neighbourhood
                int nFlips2 =
                    acq::orientCloudNormals(
                        /* [in    ] Lists of neighbours: */ neighbours2,
                        /* [in,out]   Normals to change: */ cloudManager.getCloud(2).getNormals()
                    );
                std::cout << "nFlips: " << nFlips2 << "/" << cloudManager.getCloud(2).getNormals().size() << "\n";

                // construct the sum of both 
                 cloudManager.setCloud(acq::DecoratedCloud(cloudManager.getCloud(0),cloudManager.getCloud(2), false),3) ;

                // Update viewer
                acq::setViewerNormals(
                    /* [in, out] Viewer to update: */ viewer,
                    /* [in]            Pointcloud: */ cloudManager.getCloud(3).getVertices(),
                    /* [in] Normals of Pointcloud: */ cloudManager.getCloud(3).getNormals()
                );
            } //...lambda to call on buttonclick
        ); //...addButton(orientFLANN)

        // Add a button for estimating normals using faces as neighbourhood
        viewer.ngui->addButton(
            /* Displayed label: */ "Estimate normals (from faces)",

            /* Lambda to call: */ [&]() {
                // for the first mesh 
                // Check, if normals already exist
                acq::DecoratedCloud & cloud0 = cloudManager.getCloud(0 ) ;

                if (!cloud0.hasNormals())
                    cloud0.setNormals(
                        acq::recalcNormals(
                            /* [in]      K-neighbours for FLANN: */ kNeighbours,
                            /* [in]             Vertices matrix: */ cloud0.getVertices(),
                            /* [in]      max neighbour distance: */ maxNeighbourDist
                        )
                    );

                // Estimate neighbours using FLANN
                acq::NeighboursT const neighbours =
                    acq::calculateCloudNeighboursFromFaces(
                        /* [in] Faces: */ cloud0.getFaces()
                    );

                // Estimate normals for points in cloud vertices
                cloud0.setNormals(
                    acq::calculateCloudNormals(
                        /* [in]               Cloud: */ cloud0.getVertices(),
                        /* [in] Lists of neighbours: */ neighbours
                    )
                );


                // for the second mesh 
                if (!cloudManager.getCloud(2).hasNormals())
                    cloudManager.getCloud(2).setNormals(
                        acq::recalcNormals(
                            /* [in]      K-neighbours for FLANN: */ kNeighbours,
                            /* [in]             Vertices matrix: */ cloudManager.getCloud(2).getVertices(),
                            /* [in]      max neighbour distance: */ maxNeighbourDist
                        )
                    );

                // Estimate neighbours using FLANN
                acq::NeighboursT const neighbours2 =
                    acq::calculateCloudNeighboursFromFaces(
                        /* [in] Faces: */ cloudManager.getCloud(2).getFaces()
                    );

                // Estimate normals for points in cloud vertices
                cloudManager.getCloud(2).setNormals(
                    acq::calculateCloudNormals(
                        /* [in]               Cloud: */ cloudManager.getCloud(2).getVertices(),
                        /* [in] Lists of neighbours: */ neighbours2
                    )
                );

                // sum of both 
                 cloudManager.setCloud(acq::DecoratedCloud(cloudManager.getCloud(0), cloudManager.getCloud(2), false),3) ;

                // Update viewer
                acq::setViewerNormals(
                    /* [in, out] Viewer to update: */ viewer,
                    /* [in]            Pointcloud: */ cloudManager.getCloud(3).getVertices(),
                    /* [in] Normals of Pointcloud: */ cloudManager.getCloud(3).getNormals()
                );
            } //...button push lambda
        ); //...estimate normals from faces

        // Add a button for orienting normals using face information
        viewer.ngui->addButton(
            /* Displayed label: */ "Orient normals (from faces)",

            /* Lambda to call: */ [&]() {
                // for the first mesh 
                // Check, if normals already exist
                if (!cloudManager.getCloud(0).hasNormals())
                    cloudManager.getCloud(0).setNormals(
                        acq::recalcNormals(
                            /* [in]      K-neighbours for FLANN: */ kNeighbours,
                            /* [in]             Vertices matrix: */ cloudManager.getCloud(0).getVertices(),
                            /* [in]      max neighbour distance: */ maxNeighbourDist
                        )
                    );

                // Orient normals in place using established neighbourhood
                int nFlips =
                    acq::orientCloudNormalsFromFaces(
                        /* [in    ] Lists of neighbours: */ cloudManager.getCloud(0).getFaces(),
                        /* [in,out]   Normals to change: */ cloudManager.getCloud(0).getNormals()
                    );
                std::cout << "nFlips: " << nFlips << "/" << cloudManager.getCloud(0).getNormals().size() << "\n";

                // for the seocnd mesh 
                if (!cloudManager.getCloud(2).hasNormals())
                    cloudManager.getCloud(2).setNormals(
                        acq::recalcNormals(
                            /* [in]      K-neighbours for FLANN: */ kNeighbours,
                            /* [in]             Vertices matrix: */ cloudManager.getCloud(2).getVertices(),
                            /* [in]      max neighbour distance: */ maxNeighbourDist
                        )
                    );

                // Orient normals in place using established neighbourhood
                int nFlips2 =
                    acq::orientCloudNormalsFromFaces(
                        /* [in    ] Lists of neighbours: */ cloudManager.getCloud(2).getFaces(),
                        /* [in,out]   Normals to change: */ cloudManager.getCloud(2).getNormals()
                    );
                std::cout << "nFlips: " << nFlips2 << "/" << cloudManager.getCloud(2).getNormals().size() << "\n";

                // sum of both 
                 cloudManager.setCloud(acq::DecoratedCloud(cloudManager.getCloud(0), cloudManager.getCloud(2), false),3) ;

                // Update viewer
                acq::setViewerNormals(
                    /* [in, out] Viewer to update: */ viewer,
                    /* [in]            Pointcloud: */ cloudManager.getCloud(3).getVertices(),
                    /* [in] Normals of Pointcloud: */ cloudManager.getCloud(3).getNormals()
                );
            } //...lambda to call on buttonclick
        ); //...addButton(orientFromFaces)

        // Add a button for flipping normals
        viewer.ngui->addButton(
            /* Displayed label: */ "Flip normals",
            /*  Lambda to call: */ [&](){
                // Store reference to current cloud (id 0 for now)
                acq::DecoratedCloud &cloud = cloudManager.getCloud(3);

                // Flip normals
                cloud.getNormals() *= -1.f;

                // Update viewer
                acq::setViewerNormals(
                    /* [in, out] Viewer to update: */ viewer,
                    /* [in]            Pointcloud: */ cloud.getVertices(),
                    /* [in] Normals of Pointcloud: */ cloud.getNormals()
                );
            } //...lambda to call on buttonclick
        );

        // Add a button for setting estimated normals for shading
        viewer.ngui->addButton(
            /* Displayed label: */ "Set shading normals",
            /*  Lambda to call: */ [&](){

                // Store reference to current cloud (id 0 for now)
                acq::DecoratedCloud &cloud = cloudManager.getCloud(3);

                // Set normals to be used by viewer
                viewer.data.set_normals(cloud.getNormals());

            } //...lambda to call on buttonclick
        );

        // add a group for to test the ICP classic 
        viewer.ngui->addGroup("Test classic ICP");

        // Add a button to align them   
        viewer.ngui->addButton("Align two meshes",[&]() {

        // rotation matrix for the second meshes 
        double theta = 35*0.0174533 ;
        cloudManager.getCloud(2).install(theta,  -0.049,0,-0.012) ;

        // Calculate normals after transformation 
        cloudManager.getCloud(2).setNormals(
            acq::recalcNormals(
                /* [in]      K-neighbours for FLANN: */  kNeighbours,
                /* [in]             Vertices matrix: */ cloudManager.getCloud(2).getVertices(),
                /* [in]      max neighbour distance: */ maxNeighbourDist
            )
        );

        // construct the combined mesh and place it in third position 
        cloudManager.setCloud(acq::DecoratedCloud(cloudManager.getCloud(0), cloudManager.getCloud(2), false),3) ;

        // Show mesh
        viewer.data.clear() ;
        viewer.data.set_mesh(
            cloudManager.getCloud(3).getVertices(),
            cloudManager.getCloud(3).getFaces()
        );
        viewer.data.set_colors(cloudManager.getCloud(3).getColors());      
        });


        // Add a button for ICP algorithm 
        viewer.ngui->addButton("Apply ICP",[&]() {

        // fixed cloud
        acq::DecoratedCloud const &cloud1 = cloudManager.getCloud(0);

        // apply ICP to the second cloud using the kd-tree built at the begining 
        cloudManager.getCloud(2).icpAlgo(cloud1, nbP, kdTreeConst) ;

        // Recompute normals for the moved mesh 
        cloudManager.getCloud(2).setNormals(
            acq::recalcNormals(
                 kNeighbours,
                 cloudManager.getCloud(2).getVertices(),
                 maxNeighbourDist
            )
        );
        // construct the combined mesh
        cloudManager.setCloud(acq::DecoratedCloud(cloud1, cloudManager.getCloud(2), false),3) ;

        // Show mesh with color and vertices 
        viewer.data.clear() ;
        viewer.data.set_mesh(
                    cloudManager.getCloud(3).getVertices(),
                    cloudManager.getCloud(3).getFaces()
        ); 
        viewer.data.set_colors(cloudManager.getCloud(3).getColors());
        });

        // Add a button get the old cloud back  
        viewer.ngui->addButton("Back to start",[&]() {
        // compute the combined cloud from the unmodified clouds
        cloudManager.setCloud(acq::DecoratedCloud(cloudManager.getCloud(0), cloudManager.getCloud(1), false),3) ;
        cloudManager.setCloud(cloudManager.getCloud(1),2) ;
        std::cout << "Mesh to initial position" << std::endl ;

        // Show mesh
        viewer.data.clear() ;
        viewer.data.set_mesh(
                    cloudManager.getCloud(3).getVertices(),
                    cloudManager.getCloud(3).getFaces()
        ); 
        viewer.data.set_colors(cloudManager.getCloud(3).getColors());
        });

        // group to test how ICP converges 
        viewer.ngui->addGroup("How ICP converges");

        // Add a button to show overlaps after alignment   
        viewer.ngui->addButton("Show overlap",[&]() {
        // construct a new cloud allowing to set the intersections in an other color 
        cloudManager.setCloud(acq::DecoratedCloud(cloudManager.getCloud(0), cloudManager.getCloud(2), true),3) ;
        std::cout << "Overlaps computed" << std::endl ;

        // Show mesh
        viewer.data.clear() ;
        viewer.data.set_mesh(
                    cloudManager.getCloud(3).getVertices(),
                    cloudManager.getCloud(3).getFaces()
        ); 
        viewer.data.set_colors(cloudManager.getCloud(3).getColors());   
        });

        // add a variable to decide how to rotate the mesh 
        viewer.ngui->addVariable<float>( "Rotation around z axis : ",[&] (int val) {
                angle = val ; // set the value 
            }, [&]() {
                return angle; // get the value into angle 
            } 
        ); 

        // Rotate the mesh  
        viewer.ngui->addButton("Rotate on z",[&]() {

        // rotation using "angle" value 
        double theta = angle*0.0174533 ;
        // move the mesh 
        cloudManager.getCloud(2).install(theta, 0.0, 0.0, 0.0) ; 

        // Calculate normals after movement 
        cloudManager.getCloud(2).setNormals(
            acq::recalcNormals(
                kNeighbours,
                cloudManager.getCloud(2).getVertices(),
                maxNeighbourDist
            )
        );
        std::cout << "Rotation applied" << std::endl ;

        // construct the new cloud with the moved one and the fixed one
        cloudManager.setCloud(acq::DecoratedCloud(cloudManager.getCloud(0), cloudManager.getCloud(2), false),3) ;

        // Show mesh
        viewer.data.clear() ;
        viewer.data.set_mesh(
            cloudManager.getCloud(3).getVertices(),
            cloudManager.getCloud(3).getFaces()
        );
        viewer.data.set_colors(cloudManager.getCloud(3).getColors()) ;
        });

       // Add a button to add some noise   
        viewer.ngui->addButton("Add noise",[&]() {

        // add noise to the second cloud 
        cloudManager.getCloud(2).addNoise(sigmaX, sigmaY, sigmaZ) ;

        // construct the visualization
        cloudManager.setCloud(acq::DecoratedCloud(cloudManager.getCloud(0), cloudManager.getCloud(2), false),3) ;

        cloudManager.getCloud(2).setNormals(
            acq::recalcNormals(
                 kNeighbours,
                 cloudManager.getCloud(2).getVertices(),
                 maxNeighbourDist
            )
        );
        std::cout << "Noise added" << std::endl ;

        // Show mesh
        viewer.data.clear() ;
        viewer.data.set_mesh(
            cloudManager.getCloud(3).getVertices(),
            cloudManager.getCloud(3).getFaces()
        );
        viewer.data.set_colors(cloudManager.getCloud(3).getColors());        
        });

        // add a group a handle multipled meshes 
        viewer.ngui->addGroup("Multiple meshes");

        // Add a button to show overlap  
        viewer.ngui->addButton("Add meshes",[&]() {
        // Install meshes to apply ICP after 
        double theta30 = 30*0.0174533 ;
        cloudManager.getCloud(2).install(theta30,-0.05,0,-0.015) ;

        double theta90 =  90*0.0174533 ;
        cloudManager.getCloud(4).install(theta90, 0.0,0.0, 0.0) ; 

        double theta180 =  180*0.0174533 ;
        cloudManager.getCloud(5).install(theta180, 0.0,0.0,0.0) ; 

        double theta270 =  270*0.0174533 ;
        cloudManager.getCloud(6).install(theta270, 0.0,0.0,0.0);

        double theta315 =  315*0.0174533 ;
        cloudManager.getCloud(7).install(theta315, -0.01,0.0,-0.016) ;

        // Calculate normals after transformation 
        cloudManager.getCloud(4).setNormals(
            acq::recalcNormals(
                 kNeighbours,
                 cloudManager.getCloud(4).getVertices(),
                 maxNeighbourDist
            )
        );

        cloudManager.getCloud(2).setNormals(
            acq::recalcNormals(
                 kNeighbours,
                 cloudManager.getCloud(2).getVertices(),
                 maxNeighbourDist
            )
        );

      cloudManager.getCloud(5).setNormals(
            acq::recalcNormals(
                 kNeighbours,
                 cloudManager.getCloud(5).getVertices(),
                 maxNeighbourDist
            )
        );

      cloudManager.getCloud(7).setNormals(
            acq::recalcNormals(
                kNeighbours,
                cloudManager.getCloud(7).getVertices(),
                maxNeighbourDist
            )
        );

      cloudManager.getCloud(6).setNormals(
            acq::recalcNormals(
                kNeighbours,
                cloudManager.getCloud(6).getVertices(),
                maxNeighbourDist
            )
        );

        // construct a new mesh with all the meshes to visualize
        cloudManager.setCloud(acq::DecoratedCloud(cloudManager.getCloud(0), cloudManager.getCloud(2), 
                                                  cloudManager.getCloud(4),cloudManager.getCloud(5),
                                                  cloudManager.getCloud(7),cloudManager.getCloud(6)),3) ;

        // Show mesh
        viewer.data.clear() ;
        viewer.data.set_mesh(
            cloudManager.getCloud(3).getVertices(),
            cloudManager.getCloud(3).getFaces()
        );
        viewer.data.set_colors(cloudManager.getCloud(3).getColors());
        });

        // Add a button for ICP to multiple meshes 
        viewer.ngui->addButton("Apply ICP to multiple meshes",[&]() {

        // place first 3 with respect to the first mesh 
        cloudManager.getCloud(2).icpAlgo(cloudManager.getCloud(0), nbP, kdTreeConst) ;
        cloudManager.getCloud(7).icpAlgo(cloudManager.getCloud(0), nbP, kdTreeConst) ;
        cloudManager.getCloud(4).icpAlgo(cloudManager.getCloud(0), nbP, kdTreeConst) ;

        // Recompute normals for the 3 moved mesh 
        cloudManager.getCloud(4).setNormals(
            acq::recalcNormals(
                kNeighbours,
                cloudManager.getCloud(4).getVertices(),
                maxNeighbourDist
            )
        );

        cloudManager.getCloud(7).setNormals(
            acq::recalcNormals(
                kNeighbours,
                cloudManager.getCloud(7).getVertices(),
                maxNeighbourDist
            )
        );

        cloudManager.getCloud(2).setNormals(
            acq::recalcNormals(
                kNeighbours,
                cloudManager.getCloud(2).getVertices(),
                maxNeighbourDist
            )
        );

        // place the last two with respect to moved meshes 
        
        // construct a kd-tree with the 4th to place the fith mesh 
        // initialization 
        int dim2 = cloudManager.getCloud(4).getVertices().cols() ;
        int nPts2 = cloudManager.getCloud(4).getVertices().rows() ;
        ANNkd_tree*	kdTreeConst2 ;
        ANNpointArray vertices2Match2 ; 
        vertices2Match2 = annAllocPts(nPts2, dim2);

        // construct an array from the vertices matrix
        vertices2Match2 = matrixToANNArray(cloudManager.getCloud(4).getVertices()) ; 
        
        // construct the tree 
        kdTreeConst2 = new ANNkd_tree(			
					vertices2Match2,		
					nPts2,			
					dim2);		
        
        // apply ICP with the new tree with respect to the 4th mesh  
        cloudManager.getCloud(5).icpAlgo(cloudManager.getCloud(4), nbP, kdTreeConst2) ;

        // free the memory of this tree after use 
        delete kdTreeConst2;

        // construct an other tree to place the 6th mesh   
        int dim3 = cloudManager.getCloud(7).getVertices().cols() ;
        int nPts3 = cloudManager.getCloud(7).getVertices().rows() ;
        ANNkd_tree*	kdTreeConst3 ;
        ANNpointArray vertices2Match3 ; // array of points 
        vertices2Match3 = annAllocPts(nPts3, dim3);

        vertices2Match3 = matrixToANNArray(cloudManager.getCloud(7).getVertices()) ; 
        
        kdTreeConst3 = new ANNkd_tree(					// build search structure
					vertices2Match3,					// the data points
					nPts3,						// number of points
					dim3);						// dimension of space
        
        // apply ICP with the new tree with respect to the 7th mesh 
        cloudManager.getCloud(6).icpAlgo(cloudManager.getCloud(7), nbP, kdTreeConst3) ;
        // free the memory of this tree after use 
        delete kdTreeConst3;

        // Recompute normals for the moved meshes
        cloudManager.getCloud(5).setNormals(
            acq::recalcNormals(
                kNeighbours,
                cloudManager.getCloud(5).getVertices(),
                maxNeighbourDist
            )
        );

        cloudManager.getCloud(6).setNormals(
            acq::recalcNormals(
                kNeighbours,
                cloudManager.getCloud(6).getVertices(),
                maxNeighbourDist
            )
        );

        // compute the combined mesh 
        cloudManager.setCloud(acq::DecoratedCloud(cloudManager.getCloud(0), cloudManager.getCloud(2), 
                                                  cloudManager.getCloud(4), cloudManager.getCloud(5),
                                                  cloudManager.getCloud(7), cloudManager.getCloud(6)),3) ;
        // Show mesh
        viewer.data.clear() ;
        viewer.data.set_mesh(
                    cloudManager.getCloud(3).getVertices(),
                    cloudManager.getCloud(3).getFaces()
        ); 
        viewer.data.set_colors(cloudManager.getCloud(3).getColors());
        });

        // Add new group for ICP point to plane 
        viewer.ngui->addGroup("ICP point to plane");

        viewer.ngui->addButton("Apply ICP points to plane",[&]() {
        // apply ICP point to plane 
        cloudManager.getCloud(2).icpPointToPlane(cloudManager.getCloud(0), nbP, kdTreeConst) ;

        // Recompute normals for the moved mesh 
        cloudManager.getCloud(2).setNormals(
            acq::recalcNormals(
                kNeighbours,
                cloudManager.getCloud(2).getVertices(),
                maxNeighbourDist
            )
        );

        cloudManager.setCloud(acq::DecoratedCloud(cloudManager.getCloud(0), cloudManager.getCloud(2), false),3) ;

        // Show mesh
        viewer.data.clear() ;
        viewer.data.set_mesh(
                    cloudManager.getCloud(3).getVertices(),
                    cloudManager.getCloud(3).getFaces()
        ); 
        viewer.data.set_colors(cloudManager.getCloud(3).getColors());
        });

        // Generate menu
        viewer.screen->performLayout();

        return false;
    }; //...viewer menu


    // Start viewer
    viewer.launch();

    return 0;
} //...main()


