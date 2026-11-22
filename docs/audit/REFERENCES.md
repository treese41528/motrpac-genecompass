# References for the methods audit (2026-11-11)

Every reference named in the nine audit parts, with how it was checked.

**How they were checked.**
- **Crossref.** Each DOI below was looked up on `api.crossref.org` on 2026-11-11. Title, authors, journal, year,
  volume, issue and pages are printed from the Crossref record, not typed from memory. That covers 81 entries:
  the manuscript's 27 (which reproduce its 2026-11-04 check) and 54 that the audit adds. Four further lookups,
  for tools the pipeline turned out not to call, are in the raw record only.
- **arXiv and JMLR.** Four preprints were checked against the arXiv record and one paper against its JMLR page.
- **Paper PDFs in `readings/`.** Two statements were checked in the paper itself:
  - Chu 2022: the expression-accuracy benchmark (> 0.95 for tumours with > 50% purity, Fig. 1h; Pearson on
    variance-stabilised values).
  - Yang 2024 (GeneCompass): the corpus QC rules, the 2,048-gene sequence, and the construction of the four
    priors (Methods, PDF pages 12-13).
- **Not verified.** Four entries are marked "cited from memory, not verified" or "no registry record checked":
  two books, one workshop paper and one R package. No registry was consulted for them.

**What the check does and does not establish.** It establishes that each paper exists with the stated
bibliographic details. Except for the two PDF checks above, it does not establish that a paper says what the
code or a project document attributes to it.

**Not cited anywhere in the repository.** Many methods the pipeline uses have no citation in the code, the docs
or the manuscript. The audit parts mark each one; the entries here are the references a methods section would
need. The manuscript bibliography has 27 entries.

Raw lookup output: `docs/audit/crossref_record.tsv`.

## The study and its biology

| Reference | DOI or identifier | Used for (audit part) |
|---|---|---|
| Sanford JA, Nogiec CD, Lindholm ME, Adkins JN, et al. Molecular Transducers of Physical Activity Consortium (MoTrPAC): Mapping the Dynamic Responses to Exercise. *Cell* 2020;181(7):1464-1474. | `10.1016/j.cell.2020.06.004` | MoTrPAC design (background) |
| MoTrPAC Study Group. Temporal dynamics of the multi-omic response to endurance exercise training. *Nature* 2024;629(8010):174-183. | `10.1038/s41586-023-06877-w` | The rat training study and its bulk data (4, 7) |
| Nair VD, Pincas H, Smith GR, Zaslavsky E, et al. Molecular adaptations in response to exercise training are associated with tissue-specific transcriptomic and epigenomic signatures. *Cell Genomics* 2024;4(6):100421. | `10.1016/j.xgen.2023.100421` | Tissue-specific transcriptomic signatures (background) |
| Many GM, Sanford JA, Sagendorf TJ, Hou Z, et al. Sexual dimorphism and the multi-omic response to exercise training in rat subcutaneous white adipose tissue. *Nature Metabolism* 2024;6(5):963-979. | `10.1038/s42255-023-00959-9` | White adipose response (background) |
| Vetr NG, Gay NR, MoTrPAC Study Group, Adkins JN, et al. The impact of exercise on gene regulation in association with complex trait genetics. *Nature Communications* 2024;15(1):3346. | `10.1038/s41467-024-45966-w` | DE recipe followed; positive-control gene programs (7) |
| Yu Y, Zhang X, Chen Y, Li Y, et al. Single-cell sequencing of immune cells after marathon and symptom-limited cardiopulmonary exercise. *iScience* 2023;26(4):106532. | `10.1016/j.isci.2023.106532` | Human immune positive controls (7) |
| Ma S, Sun S, Geng L, Song M, et al. Caloric Restriction Reprograms the Single-Cell Transcriptional Landscape of Rattus Norvegicus Aging. *Cell* 2020;180(5):984-1001.e22. | `10.1016/j.cell.2020.02.008` | Source study of the WATSC, MUSCLE and BAT cross-check references, GSE137869 (4) |
| Chow LS, Gerszten RE, Taylor JM, Pedersen BK, et al. Exerkines in health, resilience and disease. *Nature Reviews Endocrinology* 2022;18(5):273-289. | `10.1038/s41574-022-00641-2` | Exerkines (background) |
| Söllner JF, Leparc G, Hildebrandt T, Klein H, et al. An RNA-Seq atlas of gene expression in mouse and rat normal tissues. *Scientific Data* 2017;4(1):170185. | `10.1038/sdata.2017.185` | Rat and mouse tissue atlas (background) |
| Kelahmetoglu Y, Jannig PR, Cervenka I, Koch LG, et al. Comparative Analysis of Skeletal Muscle Transcriptional Signatures Associated With Aerobic Exercise Capacity or Response to Training in Humans and Rats. *Frontiers in Endocrinology* 2020;11:591476. | `10.3389/fendo.2020.591476` | Human-rat muscle comparison (background) |

## Models and architecture

| Reference | DOI or identifier | Used for (audit part) |
|---|---|---|
| Yang X, Liu G, Feng G, Bu D, et al. GeneCompass: deciphering universal gene regulatory mechanisms with a knowledge-informed cross-species foundation model. *Cell Research* 2024;34(12):830-845. | `10.1038/s41422-024-01034-y` | GeneCompass: the base model, tokenization, the four priors, corpus QC rules (1, 2, 3) |
| Theodoris CV, Xiao L, Chopra A, Chaffin MD, et al. Transfer learning enables predictions in network biology. *Nature* 2023;618(7965):616-624. | `10.1038/s41586-023-06139-9` | Geneformer: rank-value encoding and in-silico deletion (2, 3, 9) |
| Devlin J, Chang MW, Lee K, Toutanova K BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)* 2019:4171-4186. | `10.18653/v1/N19-1423` | Encoder architecture and the 15% / 80-10-10 masking rule (3) |
| Wolf T, Debut L, Sanh V, Chaumond J, et al. Transformers: State-of-the-Art Natural Language Processing. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations* 2020:38-45. | `10.18653/v1/2020.emnlp-demos.6` | Hugging Face `transformers` Trainer (3) |
| Loshchilov I, Hutter F. Decoupled Weight Decay Regularization. ICLR 2019. | `arXiv:1711.05101` (arXiv record checked) | AdamW optimiser (3) |
| Ji Y, Zhou Z, Liu H, Davuluri RV DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome. *Bioinformatics* 2021;37(15):2112-2120. | `10.1093/bioinformatics/btab083` | DNABERT, the promoter encoder (2) |
| Du J, Jia P, Dai Y, Tao C, et al. Gene2vec: distributed representation of genes based on co-expression. *BMC Genomics* 2019;20(Suppl 1):82. | `10.1186/s12864-018-5370-x` | gene2vec, used for the co-expression, family and GRN priors (2) |
| Mikolov T, Chen K, Corrado G, Dean J. Efficient Estimation of Word Representations in Vector Space. 2013. | `arXiv:1301.3781` (arXiv record checked) | Word2Vec skip-gram behind gene2vec (2) |
| Rehurek R, Sojka P. Software Framework for Topic Modelling with Large Corpora. LREC 2010 Workshop on New Challenges for NLP Frameworks. | none found; **cited from memory, not verified** | gensim implementation (2) |
| Duren Z, Chen X, Jiang R, Wang Y, et al. Modeling gene regulation from paired expression and chromatin accessibility data. *Proceedings of the National Academy of Sciences* 2017;114(25):E4914-E4923. | `10.1073/pnas.1704553114` | PECA, the method behind GeneCompass's GRN prior (2). Whether this paper describes the PECA2 version GeneCompass used was not checked |
| Lambert SA, Jolma A, Campitelli LF, Das PK, et al. The Human Transcription Factors. *Cell* 2018;172(4):650-665. | `10.1016/j.cell.2018.01.029` | Human transcription-factor list, the source of the rat TF list (9) |
| Kamimoto K, Stringa B, Hoffmann CM, Jindal K, et al. Dissecting cell identity via network inference and in silico gene perturbation. *Nature* 2023;614(7949):742-751. | `10.1038/s41586-022-05688-9` | CellOracle (related work on in-silico perturbation, 9) |
| Lotfollahi M, Klimovskaia Susmelj A, De Donno C, Hetzel L, et al. Predicting cellular responses to complex perturbations in high‐throughput screens. *Molecular Systems Biology* 2023;19(6):e11517. | `10.15252/msb.202211517` | CPA (Module C; environment only, nothing run; 9) |
| Cui H, Wang C, Maan H, Pang K, et al. scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods* 2024;21(8):1470-1480. | `10.1038/s41592-024-02201-0` | scGPT (related work) |
| Hao M, Gong J, Zeng X, Liu C, et al. Large-scale foundation model on single-cell transcriptomics. *Nature Methods* 2024;21(8):1481-1491. | `10.1038/s41592-024-02305-7` | scFoundation; named in Stage 2's reverse-normalisation comment (1) |

## Data sources and annotation resources

| Reference | DOI or identifier | Used for (audit part) |
|---|---|---|
| Barrett T, Wilhite SE, Ledoux P, Evangelista C, et al. NCBI GEO: archive for functional genomics data sets—update. *Nucleic Acids Research* 2013;41(D1):D991-D995. | `10.1093/nar/gks1193` | GEO harvest (1) |
| Sarkans U, Füllgrabe A, Ali A, Athar A, et al. From ArrayExpress to BioStudies. *Nucleic Acids Research* 2021;49(D1):D1502-D1506. | `10.1093/nar/gkaa1062` | ArrayExpress / BioStudies harvest (1); not cited in the repository |
| Dyer SC, Austine-Orimoloye O, Azov AG, Barba M, et al. Ensembl 2025. *Nucleic Acids Research* 2025;53(D1):D948-D957. | `10.1093/nar/gkae1071` | Ensembl release 115 gene set and orthology (1, 2, 4); not cited in the repository |
| Kinsella RJ, Kahari A, Haider S, Zamora J, et al. Ensembl BioMarts: a hub for data retrieval across taxonomic space. *Database* 2011;2011:bar030. | `10.1093/database/bar030` | BioMart queries (1, 2); not cited in the repository |
| Vedi M, Smith JR, Thomas Hayman G, Tutaj M, et al. 2022 updates to the Rat Genome Database: a Findable, Accessible, Interoperable, and Reusable (FAIR) resource. *GENETICS* 2023;224(1):iyad042. | `10.1093/genetics/iyad042` | RGD symbol synonyms (1, 4); not cited in the repository |
| Seal RL, Braschi B, Gray K, Jones TEM, et al. Genenames.org: the HGNC resources in 2023. *Nucleic Acids Research* 2023;51(D1):D1003-D1009. | `10.1093/nar/gkac888` | HGNC gene families (2); not cited in the repository |
| Cock PJA, Antao T, Chang JT, Chapman BA, et al. Biopython: freely available Python tools for computational molecular biology and bioinformatics. *Bioinformatics* 2009;25(11):1422-1423. | `10.1093/bioinformatics/btp163` | Entrez access through Biopython (1) |
| Franzén O, Gan LM, Björkegren JLM PanglaoDB: a web server for exploration of mouse and human single-cell RNA sequencing data. *Database* 2019;2019:baz046. | `10.1093/database/baz046` | PanglaoDB marker sets for corpus cell annotation (3, 4) |
| Ianevski A, Giri AK, Aittokallio T Fully-automated and ultra-fast cell-type identification using specific marker combinations from single-cell transcriptomic data. *Nature Communications* 2022;13(1):1246. | `10.1038/s41467-022-28803-w` | ScType scoring, re-implemented in `analysis/annotate_sctype.py` (3, 4) |
| The Tabula Sapiens Consortium. The Tabula Sapiens: A multiple-organ, single-cell transcriptomic atlas of humans. *Science* 2022;376(6594):eabl4896. | `10.1126/science.abl4896` | Tabula Sapiens atlas (8) |
| CZI Cell Science Program, Abdulla S, Aevermann B, Assis P, et al. CZ CELLxGENE Discover: a single-cell data platform for scalable exploration, analysis and modeling of aggregated data. *Nucleic Acids Research* 2025;53(D1):D886-D900. | `10.1093/nar/gkae1142` | CELLxGENE, where the atlas files were retrieved (8); not cited in the repository |
| Shen WK, Chen SY, Gan ZQ, Zhang YZ, et al. AnimalTFDB 4.0: a comprehensive animal transcription factor database updated with variation and expression annotations. *Nucleic Acids Research* 2023;51(D1):D39-D45. | `10.1093/nar/gkac907` | Named in two project documents as the TF source; only the Lambert list is on disk (9) |

## Single-cell processing

| Reference | DOI or identifier | Used for (audit part) |
|---|---|---|
| Wolf FA, Angerer P, Theis FJ SCANPY: large-scale single-cell gene expression data analysis. *Genome Biology* 2018;19(1):15. | `10.1186/s13059-017-1382-0` | Scanpy (1, 3, 4) |
| Virshup I, Rybakov S, Theis FJ, Angerer P, et al. anndata: Access and store annotated data matrices. *Journal of Open Source Software* 2024;9(101):4371. | `10.21105/joss.04371` | AnnData files throughout |
| Virtanen P, Gommers R, Oliphant TE, Haberland M, et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods* 2020;17(3):261-272. | `10.1038/s41592-019-0686-2` | SciPy (`zscore`, `wilcoxon`, `spearmanr`) |
| Traag VA, Waltman L, van Eck NJ From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports* 2019;9(1):5233. | `10.1038/s41598-019-41695-z` | Leiden clustering (3, 4) |
| Stuart T, Butler A, Hoffman P, Hafemeister C, et al. Comprehensive Integration of Single-Cell Data. *Cell* 2019;177(7):1888-1902.e21. | `10.1016/j.cell.2019.05.031` | `seurat_v3` highly-variable-gene selection (3, 4) |
| Korsunsky I, Millard N, Fan J, Slowikowski K, et al. Fast, sensitive and accurate integration of single-cell data with Harmony. *Nature Methods* 2019;16(12):1289-1296. | `10.1038/s41592-019-0619-0` | Harmony integration in the joint re-annotation (4) |
| Young MD, Behjati S SoupX removes ambient RNA contamination from droplet-based single-cell RNA sequencing data. *GigaScience* 2020;9(12):giaa151. | `10.1093/gigascience/giaa151` | SoupX ambient-RNA correction attempts on HIPPOC (4) |
| McInnes L, Healy J, Melville J. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. 2018. | `arXiv:1802.03426` (arXiv record checked) | UMAP viewer (3, 6) |

## Deconvolution and its benchmarks

| Reference | DOI or identifier | Used for (audit part) |
|---|---|---|
| Chu T, Wang Z, Pe’er D, Danko CG Cell type and gene expression deconvolution with BayesPrism enables Bayesian integrative analysis across bulk and single-cell RNA sequencing in oncology. *Nature Cancer* 2022;3(4):505-517. | `10.1038/s43018-022-00356-3` | BayesPrism; the expression-accuracy benchmark, checked in the PDF: correlation > 0.95 for tumours with > 50% purity, Fig. 1h (5) |
| Dietrich A, Merotto L, Pelz K, Eder B, et al. omnideconv: a unifying framework for using and benchmarking single-cell-informed deconvolution of bulk RNA-seq data. *Genome Biology* 2026;27(1):6. | `10.1186/s13059-026-03955-w` | omnideconv and the benchmark whose mRNA-bias finding is tested (5) |
| Dietrich A, Sturm G, Merotto L, Marini F, et al. SimBu: bias-aware simulation of bulk RNA-seq data with variable cell-type composition. *Bioinformatics* 2022;38(Suppl 2):ii141-ii147. | `10.1093/bioinformatics/btac499` | SimBu simulator (5) |
| Newman AM, Steen CB, Liu CL, Gentles AJ, et al. Determining cell type abundance and expression from bulk tissues with digital cytometry. *Nature Biotechnology* 2019;37(7):773-782. | `10.1038/s41587-019-0114-2` | CIBERSORTx (5) |
| Tsoucas D, Dong R, Chen H, Zhu Q, et al. Accurate estimation of cell-type composition from gene expression data. *Nature Communications* 2019;10(1):2975. | `10.1038/s41467-019-10802-z` | DWLS (5) |
| Finak G, McDavid A, Yajima M, Deng J, et al. MAST: a flexible statistical framework for assessing transcriptional changes and characterizing heterogeneity in single-cell RNA sequencing data. *Genome Biology* 2015;16(1):278. | `10.1186/s13059-015-0844-5` | MAST, used by DWLS's signature step (5) |
| Wang X, Park J, Susztak K, Zhang NR, et al. Bulk tissue cell type deconvolution with multi-subject single-cell expression reference. *Nature Communications* 2019;10(1):380. | `10.1038/s41467-018-08023-x` | MuSiC (5) |
| Dong M, Thennavan A, Urrutia E, Li Y, et al. SCDC: bulk gene expression deconvolution by multiple single-cell RNA sequencing references. *Briefings in Bioinformatics* 2021;22(1):416-427. | `10.1093/bib/bbz166` | SCDC (5) |
| Jew B, Alvarez M, Rahmani E, Miao Z, et al. Accurate estimation of cell composition in bulk expression through robust integration of single-cell information. *Nature Communications* 2020;11(1):1971. | `10.1038/s41467-020-15816-6` | Bisque (5) |
| Kang K, Meng Q, Shats I, Umbach DM, et al. CDSeq: A novel complete deconvolution method for dissecting heterogeneous samples using gene expression data. *PLOS Computational Biology* 2019;15(12):e1007510. | `10.1371/journal.pcbi.1007510` | CDSeq reference-free check of BLOOD (4, 5) |
| Love MI, Huber W, Anders S Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. *Genome Biology* 2014;15(12):550. | `10.1186/s13059-014-0550-8` | DESeq2 variance-stabilising transformation in the purity-sweep scoring (5); MoTrPAC's own DE (7) |
| Charytonowicz D, Brody R, Sebra R Interpretable and context-free deconvolution of multi-scale whole transcriptomic data with UniCell deconvolve. *Nature Communications* 2023;14(1):1350. | `10.1038/s41467-023-36961-8` | UniCell Deconvolve (planned secondary method; not run) |
| Sun D, Guan X, Moran AE, Wu LY, et al. Identifying phenotype-associated subpopulations by integrating bulk and single-cell sequencing data. *Nature Biotechnology* 2022;40(4):527-538. | `10.1038/s41587-021-01091-3` | Scissor (planned secondary method; not run) |

## Statistics

| Reference | DOI or identifier | Used for (audit part) |
|---|---|---|
| Ritchie ME, Phipson B, Wu D, Hu Y, et al. limma powers differential expression analyses for RNA-sequencing and microarray studies. *Nucleic Acids Research* 2015;43(7):e47. | `10.1093/nar/gkv007` | limma (7) |
| Smyth GK Linear Models and Empirical Bayes Methods for Assessing Differential Expression in Microarray Experiments. *Statistical Applications in Genetics and Molecular Biology* 2004;3(1):Article 3. | `10.2202/1544-6115.1027` | Empirical-Bayes moderated statistics (7) |
| Law CW, Chen Y, Shi W, Smyth GK voom: precision weights unlock linear model analysis tools for RNA-seq read counts. *Genome Biology* 2014;15(2):R29. | `10.1186/gb-2014-15-2-r29` | The limma-trend / voom paper (7) |
| Phipson B, Lee S, Majewski IJ, Alexander WS, et al. Robust hyperparameter estimation protects against hypervariable genes and improves power to detect differential expression. *The Annals of Applied Statistics* 2016;10(2):946-963. | `10.1214/16-AOAS920` | Robust empirical Bayes, `robust = TRUE` (7) |
| Robinson MD, McCarthy DJ, Smyth GK edgeR: a Bioconductor package for differential expression analysis of digital gene expression data. *Bioinformatics* 2010;26(1):139-140. | `10.1093/bioinformatics/btp616` | edgeR `cpm` (7) |
| Robinson MD, Oshlack A A scaling normalization method for differential expression analysis of RNA-seq data. *Genome Biology* 2010;11(3):R25. | `10.1186/gb-2010-11-3-r25` | TMM normalisation: used by MoTrPAC, not by this pipeline (7) |
| Benjamini Y, Hochberg Y Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical Society Series B: Statistical Methodology* 1995;57(1):289-300. | `10.1111/j.2517-6161.1995.tb02031.x` | Benjamini-Hochberg FDR (6, 7, 8, 9) |
| Ignatiadis N, Klaus B, Zaugg JB, Huber W Data-driven hypothesis weighting increases detection power in genome-scale multiple testing. *Nature Methods* 2016;13(7):577-580. | `10.1038/nmeth.3885` | Independent hypothesis weighting (7) |
| Ignatiadis N, Huber W Covariate Powered Cross-Weighted Multiple Testing. *Journal of the Royal Statistical Society Series B: Statistical Methodology* 2021;83(4):720-751. | `10.1111/rssb.12411` | IHW with cross-weighting, the method the installed package implements (7) |
| Fisher RA. Statistical Methods for Research Workers. Oliver and Boyd, 1925 (combined probability test added in later editions). | book; **cited from memory, not verified** | Fisher's method for combining the two sexes' p-values (7) |
| Heller R, Yekutieli D Replicability analysis for genome-wide association studies. *The Annals of Applied Statistics* 2014;8(1):481-498. | `10.1214/13-AOAS697` | Replicability analysis behind repfdr (7) |
| Heller R, Yaacoby S, Yekutieli D repfdr: a tool for replicability analysis for genome-wide association studies. *Bioinformatics* 2014;30(20):2971-2972. | `10.1093/bioinformatics/btu434` | repfdr package (7) |
| Squair JW, Gautier M, Kathe C, Anderson MA, et al. Confronting false discoveries in single-cell differential expression. *Nature Communications* 2021;12(1):5692. | `10.1038/s41467-021-25960-2` | Pseudobulk DE justification cited in the DE code (7) |
| Skinnider MA, Squair JW, Kathe C, Anderson MA, et al. Cell type prioritization in single-cell data. *Nature Biotechnology* 2021;39(1):30-34. | `10.1038/s41587-020-0605-1` | Augur (6) |
| Squair JW, Skinnider MA, Gautier M, Foster LJ, et al. Prioritization of cell types responsive to biological perturbations in single-cell data with Augur. *Nature Protocols* 2021;16(8):3836-3873. | `10.1038/s41596-021-00561-x` | Augur protocol (6) |
| Breiman L Random Forests. *Machine Learning* 2001;45(1):5-32. | `10.1023/A:1010933404324` | Random forest, Augur's classifier (6) |
| Ojala M, Garriga GC. Permutation Tests for Studying Classifier Performance. *J Mach Learn Res* 2010;11:1833-1863. | no DOI; JMLR record checked (`jmlr.org/papers/v11/ojala10a.html`) | Permutation test of a cross-validated classifier (6, 8) |
| Phipson B, Smyth GK Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn. *Statistical Applications in Genetics and Molecular Biology* 2010;9(1):Article 39. | `10.2202/1544-6115.1585` | Permutation p-values with the +1 correction (6) |
| Westerhuis JA, Hoefsloot HCJ, Smit S, Vis DJ, et al. Assessment of PLSDA cross validation. *Metabolomics* 2008;4(1):81-89. | `10.1007/s11306-007-0099-6` | PLS-DA cross-validation pitfalls (6) |
| Westfall PH, Young SS. Resampling-Based Multiple Testing. Wiley, 1993. | book; **cited from memory, not verified** | Max-statistic permutation correction named in `EMBEDDING_DE_STANDARDS.md` (6) |
| Anderson MJ A new method for non‐parametric multivariate analysis of variance. *Austral Ecology* 2001;26(1):32-46. | `10.1111/j.1442-9993.2001.01070.pp.x` | Permutational multivariate analysis of variance, the trace statistic (6) |
| Wilcoxon F Individual Comparisons by Ranking Methods. *Biometrics Bulletin* 1945;1(6):80-83. | `10.2307/3001968` | Wilcoxon signed-rank test (3, 6) |
| Rousseeuw PJ Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics* 1987;20:53-65. | `10.1016/0377-0427(87)90125-7` | Silhouette (3, 6) |
| Hubert L, Arabie P Comparing partitions. *Journal of Classification* 1985;2(1):193-218. | `10.1007/BF01908075` | Adjusted Rand index (3) |
| Efron B Bootstrap Methods: Another Look at the Jackknife. *The Annals of Statistics* 1979;7(1):1-26. | `10.1214/aos/1176344552` | Bootstrap, used for GRN edge uncertainty (9) |
| Bendidi I, Whitfield S, Kenyon-Dean K, Ben Yedder H, El Mesbahi Y, et al. Benchmarking Transcriptomics Foundation Models for Perturbation Analysis: one PCA still rules them all. 2024. | `arXiv:2410.13956` (arXiv record checked) | The PCA baseline cited in `augur_prep.py` (6) |

## Enrichment

| Reference | DOI or identifier | Used for (audit part) |
|---|---|---|
| Subramanian A, Tamayo P, Mootha VK, Mukherjee S, et al. Gene set enrichment analysis: A knowledge-based approach for interpreting genome-wide expression profiles. *Proceedings of the National Academy of Sciences* 2005;102(43):15545-15550. | `10.1073/pnas.0506580102` | Gene set enrichment analysis (9) |
| Korotkevich G, Sukhov V, Budin N, Shpak B, et al. Fast gene set enrichment analysis. 2016 (bioRxiv; revised 2021). | `10.1101/060012` | fgsea (9) |
| Liberzon A, Birger C, Thorvaldsdóttir H, Ghandi M, et al. The Molecular Signatures Database Hallmark Gene Set Collection. *Cell Systems* 2015;1(6):417-425. | `10.1016/j.cels.2015.12.004` | MSigDB Hallmark collection (9) |
| Milacic M, Beavers D, Conley P, Gong C, et al. The Reactome Pathway Knowledgebase 2024. *Nucleic Acids Research* 2024;52(D1):D672-D678. | `10.1093/nar/gkad1025` | Reactome (9); not cited in the repository |
| Dolgalev I. msigdbr: MSigDB Gene Sets for Multiple Organisms in a Tidy Data Format. R package, version 26.1.0 installed. | software; no registry record checked | Gene-set data access (9) |

## In the manuscript bibliography, not used by any audited stage

| Reference | DOI or identifier | Used for (audit part) |
|---|---|---|
| Barbeira AN, Dickinson SP, Bonazzola R, Zheng J, et al. Exploring the phenotypic consequences of tissue specific gene expression variation inferred from GWAS summary statistics. *Nature Communications* 2018;9(1):1825. | `10.1038/s41467-018-03621-1` | S-PrediXcan (planned genetics module; unbuilt) |
