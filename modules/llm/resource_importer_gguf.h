#pragma once

#ifdef TOOLS_ENABLED

#include "core/io/resource_importer.h"

class ResourceImporterGGUF : public ResourceImporter {
	GDCLASS(ResourceImporterGGUF, ResourceImporter);

public:
	virtual String get_importer_name() const override { return "llm.gguf"; }
	virtual String get_visible_name() const override { return "GGUF Model"; }
	virtual void get_recognized_extensions(List<String> *p_extensions) const override {
		p_extensions->push_back("gguf");
	}
	// Save extension matches source — Godot won't apply further encoding.
	virtual String get_save_extension() const override { return "gguf"; }
	virtual String get_resource_type() const override { return ""; }
	virtual float get_priority() const override { return 1.0f; }
	virtual int get_import_order() const override { return 0; }
	virtual int get_preset_count() const override { return 0; }
	virtual String get_preset_name(int p_idx) const override { return String(); }
	virtual void get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset = 0) const override {}
	virtual bool get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const override { return true; }

	virtual Error import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path,
			const HashMap<StringName, Variant> &p_options,
			List<String> *r_platform_variants, List<String> *r_gen_files = nullptr,
			Variant *r_metadata = nullptr) override;
};

#endif // TOOLS_ENABLED
