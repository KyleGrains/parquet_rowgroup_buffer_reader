// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/util/io_util.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/exception.h>

#include <iostream>

using arrow::Buffer;
using arrow::Status;
using arrow::Result;

//put this in a shared_ptr
struct RowGroupBuffer
{
  char *buffer = nullptr;
  uint64_t size = 0;
  uint64_t filesize = 0;
  uint64_t fileoffset = 0;
  uint64_t row_group_index = 0;
  std::shared_ptr<parquet::FileMetaData> metadata;
  ~RowGroupBuffer()
  {
    if(buffer)
      delete[] buffer;
  }
};

/// \class RowGroupBufferReader
/// \brief Random access zero-copy reads on an arrow::Buffer
class RowGroupBufferReader
    : public arrow::io::internal::RandomAccessFileConcurrencyWrapper<RowGroupBufferReader> {
 public:
  RowGroupBufferReader(std::shared_ptr<RowGroupBuffer> row_group_info);

  bool closed() const override;

  bool supports_zero_copy() const override;

  std::shared_ptr<arrow::Buffer> buffer() const { return buffer_; }

  // Synchronous ReadAsync override
  // arrow::Future<std::shared_ptr<arrow::Buffer>> ReadAsync(const arrow::io::IOContext&, int64_t position, int64_t nbytes) override;
  //arrow::Status WillNeed(const std::vector<arrow::io::ReadRange>& ranges) override;

 protected:
  friend RandomAccessFileConcurrencyWrapper<RowGroupBufferReader>;

  Status DoClose();

  Result<int64_t> DoRead(int64_t nbytes, void* buffer);
  Result<std::shared_ptr<Buffer>> DoRead(int64_t nbytes);
  Result<int64_t> DoReadAt(int64_t position, int64_t nbytes, void* out);
  Result<std::shared_ptr<Buffer>> DoReadAt(int64_t position, int64_t nbytes);
  Result<arrow::util::string_view> DoPeek(int64_t nbytes) override;

  Result<int64_t> DoTell() const;
  Status DoSeek(int64_t position);
  Result<int64_t> DoGetSize();

  Status CheckClosed() const {
    if (!is_open_) {
      return Status::Invalid("Operation forbidden on closed RowGroupBufferReader");
    }
    return Status::OK();
  }

  std::shared_ptr<Buffer> buffer_;
  const uint8_t* data_;
  uint64_t size_;
  uint64_t filesize_;
  uint64_t position_;
  uint64_t row_group_offset_;
  bool is_open_;
};

RowGroupBufferReader::RowGroupBufferReader(std::shared_ptr<RowGroupBuffer> row_group_buffer)
    : buffer_(nullptr), data_(reinterpret_cast<uint8_t*>(row_group_buffer->buffer)), size_(row_group_buffer->size), filesize_(row_group_buffer->filesize), position_(row_group_buffer->fileoffset), row_group_offset_(row_group_buffer->fileoffset), is_open_(true) {}

Status RowGroupBufferReader::DoClose() {
  is_open_ = false;
  return Status::OK();
}

bool RowGroupBufferReader::closed() const { return !is_open_; }

Result<int64_t> RowGroupBufferReader::DoTell() const {
  RETURN_NOT_OK(CheckClosed());
  return position_;
}

Result<arrow::util::string_view> RowGroupBufferReader::DoPeek(int64_t nbytes) {
  RETURN_NOT_OK(CheckClosed());

  const int64_t bytes_available = std::min(nbytes, static_cast<int64_t>(size_ + row_group_offset_ - position_));
  return arrow::util::string_view(reinterpret_cast<const char*>(data_) + position_ - row_group_offset_,
                           static_cast<size_t>(bytes_available));
}

bool RowGroupBufferReader::supports_zero_copy() const { return false; }

Result<int64_t> ValidateReadRange(int64_t offset, int64_t size, int64_t file_size) {
  if (offset < 0 || size < 0) {
    return Status::Invalid("Invalid read (offset = ", offset, ", size = ", size, ")");
  }
  if (offset > file_size) {
    return Status::IOError("Read out of bounds (offset = ", offset, ", size = ", size,
                           ") in file of size ", file_size);
  }
  return std::min(size, file_size - offset);
}

Result<int64_t> RowGroupBufferReader::DoReadAt(int64_t position, int64_t nbytes, void* buffer) {
  RETURN_NOT_OK(CheckClosed());

  ARROW_ASSIGN_OR_RAISE(nbytes, ValidateReadRange(position - row_group_offset_, nbytes, size_));
  //ARROW_ASSIGN_OR_RAISE(nbytes, ValidateReadRange(position, nbytes, size_));
  assert(nbytes >= 0);
  if (nbytes) {
    memcpy(buffer, data_ + position - row_group_offset_, nbytes);
  }
  return nbytes;
}

Result<std::shared_ptr<Buffer>> RowGroupBufferReader::DoReadAt(int64_t position, int64_t nbytes) {
  RETURN_NOT_OK(CheckClosed());

  ARROW_ASSIGN_OR_RAISE(nbytes, ValidateReadRange(position - row_group_offset_, nbytes, size_));
  //ARROW_ASSIGN_OR_RAISE(nbytes, ValidateReadRange(position, nbytes, size_));
  assert(nbytes >= 0);

  // Arrange for data to be paged in
  // RETURN_NOT_OK(::arrow::internal::MemoryAdviseWillNeed(
  //     {{const_cast<uint8_t*>(data_ + position), static_cast<size_t>(nbytes)}}));

  if (nbytes > 0 && buffer_ != nullptr) {
    return SliceBuffer(buffer_, position - row_group_offset_, nbytes);
  } else {
    return std::make_shared<Buffer>(data_ + position - row_group_offset_, nbytes);
  }
}

Result<int64_t> RowGroupBufferReader::DoRead(int64_t nbytes, void* out) {
  RETURN_NOT_OK(CheckClosed());
  ARROW_ASSIGN_OR_RAISE(int64_t bytes_read, DoReadAt(position_, nbytes, out));
  position_ += bytes_read;
  return bytes_read;
}

Result<std::shared_ptr<Buffer>> RowGroupBufferReader::DoRead(int64_t nbytes) {
  RETURN_NOT_OK(CheckClosed());
  ARROW_ASSIGN_OR_RAISE(auto buffer, DoReadAt(position_, nbytes));
  position_ += buffer->size();
  return buffer;
}

Result<int64_t> RowGroupBufferReader::DoGetSize() {
  RETURN_NOT_OK(CheckClosed());
  //return size_;
  return filesize_;
}

Status RowGroupBufferReader::DoSeek(int64_t position) {
  RETURN_NOT_OK(CheckClosed());

  if (position - row_group_offset_ < 0 || position - row_group_offset_ > size_) {
  //if (position < 0 || position > size_) {
    return Status::IOError("Seek out of bounds");
  }

  position_ = position;
  return Status::OK();
}

struct ParquetMetadata {
  std::vector<uint64_t> rowgroup_offsets;
  std::vector<uint64_t> rowgroup_sizes;
  uint64_t num_of_rowgroups;
  uint64_t filesize;
};

ParquetMetadata fileMeta;
std::vector<std::shared_ptr<RowGroupBuffer>> rowgroup_buffers;

void read_parquet_metadata()
{
  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_ASSIGN_OR_THROW(infile,
                          arrow::io::ReadableFile::Open("parquet-arrow-example.parquet",
                                                        arrow::default_memory_pool()));

  std::unique_ptr<parquet::arrow::FileReader> reader;
  PARQUET_THROW_NOT_OK(
      parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

  PARQUET_ASSIGN_OR_THROW(fileMeta.filesize, arrow::internal::FileGetSize(infile->file_descriptor()));

  auto metadata = reader->parquet_reader()->metadata();
  fileMeta.num_of_rowgroups = metadata->num_row_groups();
  fileMeta.rowgroup_offsets.reserve(fileMeta.num_of_rowgroups);
  fileMeta.rowgroup_sizes.reserve(fileMeta.num_of_rowgroups);
  for(uint64_t i = 0; i < fileMeta.num_of_rowgroups; i++)
  {
    fileMeta.rowgroup_offsets.push_back(metadata->RowGroup(i)->file_offset());
  }

  for(uint64_t i = 0; i < fileMeta.num_of_rowgroups; i++)
  {
    uint64_t rowgroup_size = 0;
    if(i + 1 < fileMeta.num_of_rowgroups && fileMeta.rowgroup_offsets[i] > 0)
      rowgroup_size = fileMeta.rowgroup_offsets[i + 1] - fileMeta.rowgroup_offsets[i];
    if(i + 1 == fileMeta.num_of_rowgroups)
      rowgroup_size = fileMeta.filesize - fileMeta.rowgroup_offsets[i];
    fileMeta.rowgroup_sizes.push_back(rowgroup_size);
  }
}

void read_single_rowgroup_into_buffer(uint64_t row_group_index)
{
  std::shared_ptr<RowGroupBuffer> buffer = std::make_shared<RowGroupBuffer>();
  rowgroup_buffers.push_back(buffer);
  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_ASSIGN_OR_THROW(infile,
                          arrow::io::ReadableFile::Open("parquet-arrow-example.parquet",
                                                        arrow::default_memory_pool()));

  std::unique_ptr<parquet::arrow::FileReader> reader;
  PARQUET_THROW_NOT_OK(
      parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

  auto metadata = reader->parquet_reader()->metadata();

  buffer->row_group_index = row_group_index;
  buffer->size = fileMeta.rowgroup_sizes[row_group_index];
  buffer->fileoffset = fileMeta.rowgroup_offsets[row_group_index];
  buffer->filesize = fileMeta.filesize;
  buffer->buffer = new char[buffer->size];
  buffer->metadata = metadata;
  PARQUET_ASSIGN_OR_THROW(auto result , infile->ReadAt(metadata->RowGroup(row_group_index)->file_offset(), buffer->size, buffer->buffer));
  if(result != buffer->size)
    std::cerr << ("read buffer size not right");
}

void parse_single_rowgroup_from_buffer(uint64_t row_group_index)
{
  std::shared_ptr<RowGroupBuffer> buffer = rowgroup_buffers[row_group_index];
  parquet::ReaderProperties props = parquet::default_reader_properties();
  props.disable_buffered_stream();
  parquet::ArrowReaderProperties properties;

  auto bufferReader = std::make_shared<RowGroupBufferReader>(buffer);
  auto reader = parquet::ParquetFileReader::Open(bufferReader, props, buffer->metadata);
  std::unique_ptr<parquet::arrow::FileReader> filereader;
  PARQUET_THROW_NOT_OK(parquet::arrow::FileReader::Make(arrow::default_memory_pool(), std::move(reader), properties, &filereader));
  std::cout << "RowGroup index: " << buffer->row_group_index << std::endl;
  for(uint64_t i = 0; i < buffer->metadata->num_columns(); i++)
  {
    std::shared_ptr<arrow::ChunkedArray> array;
    auto column = filereader->RowGroup(buffer->row_group_index)->Column(i)->Read(&array);
    if(array)
      std::cout << array->ToString() << std::endl;
  }
  std::shared_ptr<arrow::Table> table;
  PARQUET_THROW_NOT_OK(filereader->RowGroup(buffer->row_group_index)->ReadTable(&table));
  if(table)
  {
    for(uint64_t i = 0; i < table->num_columns(); i++)
    {
      std::cout << table->column(i)->ToString() << std::endl;
    }
  }
}

int main(int argc, char** argv) {
  try{
    read_parquet_metadata();
    rowgroup_buffers.reserve(fileMeta.num_of_rowgroups);
    for(uint64_t i = 0; i < fileMeta.num_of_rowgroups; i++)
    {
      read_single_rowgroup_into_buffer(i);
      parse_single_rowgroup_from_buffer(i);
    }
  }
  catch(std::exception & e)
  {
    std::cerr << "something went wrong: " << e.what() << std::endl;
  }
}

