// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/saved_model.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "tensorflow/core/protobuf/saved_model.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace tensorflow {
class SavedModelDefaultTypeInternal : public ::google::protobuf::internal::ExplicitlyConstructed<SavedModel> {
} _SavedModel_default_instance_;

namespace protobuf_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto {


namespace {

::google::protobuf::Metadata file_level_metadata[1];

}  // namespace

PROTOBUF_CONSTEXPR_VAR ::google::protobuf::internal::ParseTableField
    const TableStruct::entries[] = {
  {0, 0, 0, ::google::protobuf::internal::kInvalidMask, 0, 0},
};

PROTOBUF_CONSTEXPR_VAR ::google::protobuf::internal::AuxillaryParseTableField
    const TableStruct::aux[] = {
  ::google::protobuf::internal::AuxillaryParseTableField(),
};
PROTOBUF_CONSTEXPR_VAR ::google::protobuf::internal::ParseTable const
    TableStruct::schema[] = {
  { NULL, NULL, 0, -1, -1, false },
};

const ::google::protobuf::uint32 TableStruct::offsets[] = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(SavedModel, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(SavedModel, saved_model_schema_version_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(SavedModel, meta_graphs_),
};

static const ::google::protobuf::internal::MigrationSchema schemas[] = {
  { 0, -1, sizeof(SavedModel)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&_SavedModel_default_instance_),
};

namespace {

void protobuf_AssignDescriptors() {
  AddDescriptors();
  ::google::protobuf::MessageFactory* factory = NULL;
  AssignDescriptors(
      "tensorflow/core/protobuf/saved_model.proto", schemas, file_default_instances, TableStruct::offsets, factory,
      file_level_metadata, NULL, NULL);
}

GOOGLE_ATTRIBUTE_NOINLINE void protobuf_AssignDescriptorsOnce() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 1);
}

}  // namespace

void TableStruct::Shutdown() {
  _SavedModel_default_instance_.Shutdown();
  delete file_level_metadata[0].reflection;
}

void TableStruct::InitDefaultsImpl() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::google::protobuf::internal::InitProtobufDefaults();
  ::tensorflow::protobuf_tensorflow_2fcore_2fprotobuf_2fmeta_5fgraph_2eproto::InitDefaults();
  _SavedModel_default_instance_.DefaultConstruct();
}

GOOGLE_ATTRIBUTE_NOINLINE void InitDefaults() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &TableStruct::InitDefaultsImpl);
}
void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] = {
      "\n*tensorflow/core/protobuf/saved_model.p"
      "roto\022\ntensorflow\032)tensorflow/core/protob"
      "uf/meta_graph.proto\"_\n\nSavedModel\022\"\n\032sav"
      "ed_model_schema_version\030\001 \001(\003\022-\n\013meta_gr"
      "aphs\030\002 \003(\0132\030.tensorflow.MetaGraphDefB1\n\030"
      "org.tensorflow.frameworkB\020SavedModelProt"
      "osP\001\370\001\001b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 255);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "tensorflow/core/protobuf/saved_model.proto", &protobuf_RegisterTypes);
  ::tensorflow::protobuf_tensorflow_2fcore_2fprotobuf_2fmeta_5fgraph_2eproto::AddDescriptors();
  ::google::protobuf::internal::OnShutdown(&TableStruct::Shutdown);
}

GOOGLE_ATTRIBUTE_NOINLINE void AddDescriptors() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;

}  // namespace protobuf_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto


// ===================================================================

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int SavedModel::kSavedModelSchemaVersionFieldNumber;
const int SavedModel::kMetaGraphsFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

SavedModel::SavedModel()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  if (GOOGLE_PREDICT_TRUE(this != internal_default_instance())) {
    protobuf_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto::InitDefaults();
  }
  SharedCtor();
  // @@protoc_insertion_point(constructor:tensorflow.SavedModel)
}
SavedModel::SavedModel(::google::protobuf::Arena* arena)
  : ::google::protobuf::Message(),
  _internal_metadata_(arena),
  meta_graphs_(arena) {
#ifdef GOOGLE_PROTOBUF_NO_STATIC_INITIALIZER
  protobuf_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto::InitDefaults();
#endif  // GOOGLE_PROTOBUF_NO_STATIC_INITIALIZER
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:tensorflow.SavedModel)
}
SavedModel::SavedModel(const SavedModel& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      meta_graphs_(from.meta_graphs_),
      _cached_size_(0) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  saved_model_schema_version_ = from.saved_model_schema_version_;
  // @@protoc_insertion_point(copy_constructor:tensorflow.SavedModel)
}

void SavedModel::SharedCtor() {
  saved_model_schema_version_ = GOOGLE_LONGLONG(0);
  _cached_size_ = 0;
}

SavedModel::~SavedModel() {
  // @@protoc_insertion_point(destructor:tensorflow.SavedModel)
  SharedDtor();
}

void SavedModel::SharedDtor() {
  ::google::protobuf::Arena* arena = GetArenaNoVirtual();
  if (arena != NULL) {
    return;
  }

}

void SavedModel::ArenaDtor(void* object) {
  SavedModel* _this = reinterpret_cast< SavedModel* >(object);
  (void)_this;
}
void SavedModel::RegisterArenaDtor(::google::protobuf::Arena* arena) {
}
void SavedModel::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* SavedModel::descriptor() {
  protobuf_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto::protobuf_AssignDescriptorsOnce();
  return protobuf_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const SavedModel& SavedModel::default_instance() {
  protobuf_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto::InitDefaults();
  return *internal_default_instance();
}

SavedModel* SavedModel::New(::google::protobuf::Arena* arena) const {
  return ::google::protobuf::Arena::CreateMessage<SavedModel>(arena);
}

void SavedModel::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.SavedModel)
  meta_graphs_.Clear();
  saved_model_schema_version_ = GOOGLE_LONGLONG(0);
}

bool SavedModel::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:tensorflow.SavedModel)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // int64 saved_model_schema_version = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(8u)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int64, ::google::protobuf::internal::WireFormatLite::TYPE_INT64>(
                 input, &saved_model_schema_version_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // repeated .tensorflow.MetaGraphDef meta_graphs = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(18u)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
                input, add_meta_graphs()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormatLite::SkipField(input, tag));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:tensorflow.SavedModel)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tensorflow.SavedModel)
  return false;
#undef DO_
}

void SavedModel::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tensorflow.SavedModel)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // int64 saved_model_schema_version = 1;
  if (this->saved_model_schema_version() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteInt64(1, this->saved_model_schema_version(), output);
  }

  // repeated .tensorflow.MetaGraphDef meta_graphs = 2;
  for (unsigned int i = 0, n = this->meta_graphs_size(); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      2, this->meta_graphs(i), output);
  }

  // @@protoc_insertion_point(serialize_end:tensorflow.SavedModel)
}

::google::protobuf::uint8* SavedModel::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.SavedModel)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // int64 saved_model_schema_version = 1;
  if (this->saved_model_schema_version() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt64ToArray(1, this->saved_model_schema_version(), target);
  }

  // repeated .tensorflow.MetaGraphDef meta_graphs = 2;
  for (unsigned int i = 0, n = this->meta_graphs_size(); i < n; i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageNoVirtualToArray(
        2, this->meta_graphs(i), deterministic, target);
  }

  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.SavedModel)
  return target;
}

size_t SavedModel::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.SavedModel)
  size_t total_size = 0;

  // repeated .tensorflow.MetaGraphDef meta_graphs = 2;
  {
    unsigned int count = this->meta_graphs_size();
    total_size += 1UL * count;
    for (unsigned int i = 0; i < count; i++) {
      total_size +=
        ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
          this->meta_graphs(i));
    }
  }

  // int64 saved_model_schema_version = 1;
  if (this->saved_model_schema_version() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int64Size(
        this->saved_model_schema_version());
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void SavedModel::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:tensorflow.SavedModel)
  GOOGLE_DCHECK_NE(&from, this);
  const SavedModel* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const SavedModel>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:tensorflow.SavedModel)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:tensorflow.SavedModel)
    MergeFrom(*source);
  }
}

void SavedModel::MergeFrom(const SavedModel& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.SavedModel)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  meta_graphs_.MergeFrom(from.meta_graphs_);
  if (from.saved_model_schema_version() != 0) {
    set_saved_model_schema_version(from.saved_model_schema_version());
  }
}

void SavedModel::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:tensorflow.SavedModel)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void SavedModel::CopyFrom(const SavedModel& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.SavedModel)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SavedModel::IsInitialized() const {
  return true;
}

void SavedModel::Swap(SavedModel* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    SavedModel* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == NULL) {
      delete temp;
    }
  }
}
void SavedModel::UnsafeArenaSwap(SavedModel* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void SavedModel::InternalSwap(SavedModel* other) {
  meta_graphs_.InternalSwap(&other->meta_graphs_);
  std::swap(saved_model_schema_version_, other->saved_model_schema_version_);
  std::swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata SavedModel::GetMetadata() const {
  protobuf_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto::protobuf_AssignDescriptorsOnce();
  return protobuf_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto::file_level_metadata[kIndexInFileMessages];
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// SavedModel

// int64 saved_model_schema_version = 1;
void SavedModel::clear_saved_model_schema_version() {
  saved_model_schema_version_ = GOOGLE_LONGLONG(0);
}
::google::protobuf::int64 SavedModel::saved_model_schema_version() const {
  // @@protoc_insertion_point(field_get:tensorflow.SavedModel.saved_model_schema_version)
  return saved_model_schema_version_;
}
void SavedModel::set_saved_model_schema_version(::google::protobuf::int64 value) {
  
  saved_model_schema_version_ = value;
  // @@protoc_insertion_point(field_set:tensorflow.SavedModel.saved_model_schema_version)
}

// repeated .tensorflow.MetaGraphDef meta_graphs = 2;
int SavedModel::meta_graphs_size() const {
  return meta_graphs_.size();
}
void SavedModel::clear_meta_graphs() {
  meta_graphs_.Clear();
}
const ::tensorflow::MetaGraphDef& SavedModel::meta_graphs(int index) const {
  // @@protoc_insertion_point(field_get:tensorflow.SavedModel.meta_graphs)
  return meta_graphs_.Get(index);
}
::tensorflow::MetaGraphDef* SavedModel::mutable_meta_graphs(int index) {
  // @@protoc_insertion_point(field_mutable:tensorflow.SavedModel.meta_graphs)
  return meta_graphs_.Mutable(index);
}
::tensorflow::MetaGraphDef* SavedModel::add_meta_graphs() {
  // @@protoc_insertion_point(field_add:tensorflow.SavedModel.meta_graphs)
  return meta_graphs_.Add();
}
::google::protobuf::RepeatedPtrField< ::tensorflow::MetaGraphDef >*
SavedModel::mutable_meta_graphs() {
  // @@protoc_insertion_point(field_mutable_list:tensorflow.SavedModel.meta_graphs)
  return &meta_graphs_;
}
const ::google::protobuf::RepeatedPtrField< ::tensorflow::MetaGraphDef >&
SavedModel::meta_graphs() const {
  // @@protoc_insertion_point(field_list:tensorflow.SavedModel.meta_graphs)
  return meta_graphs_;
}

#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)
