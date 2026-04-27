#ifdef TOOLS_ENABLED

#include "resource_importer_gguf.h"

#include "core/io/dir_access.h"

Error ResourceImporterGGUF::import(ResourceUID::ID p_source_id, const String &p_source_file,
		const String &p_save_path, const HashMap<StringName, Variant> &p_options,
		List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	return DirAccess::copy_absolute(p_source_file, p_save_path + ".gguf");
}

#endif // TOOLS_ENABLED
