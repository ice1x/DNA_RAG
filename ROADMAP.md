# Roadmap

## Genetic Guardrails

Improve the scientific rigor of DNA analysis beyond basic engineering safeguards (Pydantic validation, RSID format filtering, NCBI dbSNP existence check) by adding genetics-aware validation and interpretation quality controls.

| # | Task | Complexity | Description |
|---|------|------------|-------------|
| 0001 | **Population allele frequency (MAF) from dbSNP** | Low | The dbSNP API response already includes minor allele frequency data — we just don't parse it. Add `maf` and `global_maf_freq` fields to `SNPValidationResult` so interpretations can distinguish rare variants from common polymorphisms. Minimal changes to `snp_database.py`. |
| 0002 | **LLM prompt: effect size, polygenic traits, penetrance** | Low | Add instructions to the interpretation prompt telling the LLM to: report estimated effect sizes (OR) where known, flag polygenic traits where a single SNP explains little variance, and mention penetrance (carrying a variant does not guarantee the phenotype). No new APIs — purely prompt engineering. The project already has a `polygenic.py` module that can inform this. |
| 0003 | **Consumer genotyping accuracy warning** | Low | Add a static warning to the medical disclaimer noting that consumer DNA tests (23andMe, AncestryDNA, MyHeritage) have known error rates for certain variant types, especially insertions/deletions and CNVs. Single-line addition to the default `medical_disclaimer` in `config.py`. |
| 0004 | **ClinVar clinical significance lookup** | Medium | Query the NCBI ClinVar API (E-utilities) for each LLM-identified RSID and return clinical significance classification (pathogenic / likely pathogenic / VUS / likely benign / benign). Architecture mirrors the existing `SNPDatabaseClient` in `snp_database.py`. ~100 lines of new code plus tests. Lets the user see whether a variant is medically relevant or a benign polymorphism. |
| 0005 | **Penetrance data integration** | High | Penetrance (probability of developing a condition given a genotype) is disease-specific, varies by population, and has no single authoritative database. Would require curating data from ClinGen, OMIM, and primary literature. Realistically a research project, not a simple feature. |
| 0006 | **GWAS Catalog effect sizes** | High | Integrate the EBI GWAS Catalog API to attach per-SNP effect sizes (odds ratios, beta coefficients) from published genome-wide association studies. Challenging because effect sizes are study-dependent, population-specific, and a single RSID may have dozens of conflicting entries. Requires careful curation logic. |
