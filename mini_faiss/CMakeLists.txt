file(GLOB_RECURSE mini_faiss_src ${PROJECT_SOURCE_DIR}/mini_faiss/*.cc 
${PROJECT_SOURCE_DIR}/mini_faiss/*/*.cc)
message(STATUS "src file:" ${mini_faiss_src})
add_library(mini_faiss SHARED ${mini_faiss_src})

find_package(OpenMP REQUIRED)
target_link_libraries(mini_faiss PRIVATE OpenMP::OpenMP_CXX)

if(USE_MKL)
    find_package(MKL REQUIRED)
    target_link_libraries(mini_faiss PRIVATE ${MKL_LIBRARIES})
else()
    find_package(BLAS REQUIRED)
    target_link_libraries(mini_faiss PRIVATE ${BLAS_LIBRARIES})

    find_package(LAPACK REQUIRED)
    target_link_libraries(mini_faiss PRIVATE ${LAPACK_LIBRARIES})
endif()

if(USE_DOUBLE)
    add_definitions(-DUSE_DOUBLE)
endif()



