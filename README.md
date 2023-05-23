Designing CDSs for Maximizing protein expression levels
==
Input parameter
--
1. Amino Acids Aeqeunce File of the Protein
- It supports FASTA format form UniProt website, and input the full path of the file
2. N
- It is the number of total Solutions
3. G
- It is the number of Cycles(Generations)
4. The number of CDSs per Solution
5. Mutation Probability

Output File
--
- "Result.txt" file show the CDSs of the solutions. 
- "Normalized_value_quality_computation.txt" file is used to caculates the Hypervolume and the Minimum distance to ideal point.
    - "hv" execution file is used for the calculating Hypervolume


NVIDIA nvcc Compile option
--
- -use_fast_math
- --gpu-architecture=sm_89
    - This is needed to change your GPU(sm_86,sm_70...)
+ However, your GPU is needed to support the CUDA_Copperative & curand

+ If you want to change the number of threads used for Initialization kernel & Muatation kernel since your architecture, x variable is changed.