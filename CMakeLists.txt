# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
cmake_minimum_required(VERSION 3.0)
project(test_project)

add_executable(test test.cpp)
target_include_directories(test PRIVATE /home/kai/dev/arrow/cpp/build/install/include)

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set (CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

set(PARQUET_LIB_DIR "/home/kai/dev/arrow/cpp/build/install/lib")
target_link_libraries(test 
  PRIVATE /home/kai/dev/arrow/cpp/build/install/lib/libparquet.a
  PRIVATE /home/kai/dev/arrow/cpp/build/install/lib/libarrow.a 
  PRIVATE /home/kai/dev/arrow/cpp/build/install/lib/libarrow_bundled_dependencies.a 
  PRIVATE /home/kai/dev/arrow/cpp/build/jemalloc_ep-prefix/src/jemalloc_ep/lib/libjemalloc.a
)
