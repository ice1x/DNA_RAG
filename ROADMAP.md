# Roadmap

## Genetic Guardrails

Improve the scientific rigor of DNA analysis beyond basic engineering safeguards (Pydantic validation, RSID format filtering, NCBI dbSNP existence check) by adding genetics-aware validation and interpretation quality controls.

| # | Task | Complexity | Status | Description |
|---|------|------------|--------|-------------|
| 0001 | **Population allele frequency (MAF) from dbSNP** | Low | Done | Parse `global_mafs` from the dbSNP API response (prefers GnomAD, then 1000Genomes). Added `maf`, `maf_allele`, `maf_study` fields to `SNPValidationResult`. MAF is passed to the interpretation prompt so the LLM can flag common variants (>5%). |
| 0002 | **Anti-hallucination prompt & verified metadata** | Low | Done | The interpretation prompt now receives a VERIFIED DATA block with authoritative gene names, MAF, and ClinVar significance from NCBI. LLM is instructed to use only verified gene names, flag weak evidence, and not invent associations. Gene names from dbSNP overwrite LLM-claimed genes. |
| 0003 | **Consumer genotyping accuracy warning** | Low | Planned | Add a static warning to the medical disclaimer noting that consumer DNA tests (23andMe, AncestryDNA, MyHeritage) have known error rates for certain variant types, especially insertions/deletions and CNVs. Single-line addition to the default `medical_disclaimer` in `config.py`. |
| 0004 | **ClinVar clinical significance lookup** | Medium | Done | Query NCBI ClinVar (esearch + esummary) for each validated RSID. Returns germline classification (pathogenic / likely pathogenic / VUS / likely benign / benign) and associated trait name. Integrated into `SNPDatabase.validate_rsid()` with graceful degradation on failure. |
| 0005 | **Penetrance data integration** | High | Planned | Penetrance (probability of developing a condition given a genotype) is disease-specific, varies by population, and has no single authoritative database. Would require curating data from ClinGen, OMIM, and primary literature. Realistically a research project, not a simple feature. |
| 0006 | **GWAS Catalog effect sizes** | High | Planned | Integrate the EBI GWAS Catalog API to attach per-SNP effect sizes (odds ratios, beta coefficients) from published genome-wide association studies. Challenging because effect sizes are study-dependent, population-specific, and a single RSID may have dozens of conflicting entries. Requires careful curation logic. |
