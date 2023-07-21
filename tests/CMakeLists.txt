file(GLOB mini_faiss_test_src ${PROJECT_SOURCE_DIR}/tests/*.cc)

foreach(test_src ${mini_faiss_test_src})
    get_filename_component(test_filename ${test_src} NAME)
    string(REPLACE ".cc" "" test_filename ${test_filename})
    message(STATUS "found test:" ${test_filename})
    add_executable(${test_filename} ${test_src})
    target_link_libraries(${test_filename} mini_faiss)
endforeach(test_src ${mini_faiss_test_src})
