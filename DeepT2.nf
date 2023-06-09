/*
 * 'DeepT2' - A Nextflow pipeline for dectect Type 2 polyketides
 *
 * Huang Jiaquan 
 * contact: jiaquan_terry@bnu.edu.cn
 */

 /*
 * Enable DSL 2 syntax
 */
nextflow.enable.dsl = 2

/*
 *Scripts parameters
 */

params.genome     = "$PWD/genome.fa"
params.outdir     = "results"
params.prefix     = "prefix"
dataset           = "$PWD/datasets"
model             = "$PWD/model"
script            = "$PWD/scripts"  

log.info """\
DeeT2 v 1.0
================================
genome    : $params.genome
results   : $params.outdir
prefix    : $params.prefix
"""

/*
 * Process 1: Bacteria genome annotation
 */

process seqAnnotate {
    tag "annotated $genome.baseName"
    publishDir params.outdir, mode:'copy'

    input:
    path (genome)
    val prefix
    
    output:
    path (prokka_annotation)

    """
    prokka $genome --outdir prokka_annotation --prefix ${prefix} --kingdom Bacteria --rfam --cpus 12
    """
}

/*
 * Process 2: Sequence trimming and extraction
 */

process seqExtract {
    tag "Sequence trimming and extraction"

    input:
    path (prokka_annotation)
    path (dataset)
    path (script)
    val prefix
    
    output:
    path "${prefix}_hypo.faa"
    
    script:
    """
    cp ./prokka_annotation/${prefix}.faa ./
    seqkit seq -m 300 -M 500 ${prefix}.faa > ${prefix}_trim.faa
    sed -i 's/ .*//' ${prefix}_trim.faa
    hmmsearch ./datasets/ksb.hmm ${prefix}_trim.faa > ${prefix}_hmm.out
    grep -oP '(?<=>> ).*' ${prefix}_hmm.out > list.txt
    python ./scripts/fasta_extraction.py --header list.txt --input ${prefix}_trim.faa --output ${prefix}_hypo.faa
    """
}


/*
 * Process 3: Sequence embedding
 */

process seqEmbed {
    tag "Sequence embedding"
    publishDir params.outdir, mode:'copy'

    input:
    file "${prefix}_hypo.faa"
    path (script)
    val prefix
    
    output:
    path (embedding)

    """
    python ./scripts/embedding.py esm2_t36_3B_UR50D ${prefix}_hypo.faa embedding --repr_layers 36 --include mean
    """
}

/*
 * Process 4: T2PK predicting
 */

process T2PKpredict {
    tag "T2PK predicting"
    publishDir params.outdir, mode:'copy'

    input:
    path (embedding)
    file "${prefix}_hypo.faa"
    path (dataset)
    path (model)
    path (script)
    val prefix
    
    output:
    path (prediction)
    path ("log.txt")

    """
    python ./scripts/DeepT2.py --fasta ${prefix}_hypo.faa --embedding ${embedding} --output prediction --prefix $prefix 2> ${"log.txt"}
    """
}

workflow {

    // PART 1: Data preparation
    seqAnnotate(params.genome, params.prefix)
    seqExtract(seqAnnotate.out, dataset, script, params.prefix)
    seqEmbed(seqExtract.out, script, params.prefix)
    T2PKpredict(seqEmbed.out, seqExtract.out, dataset, model, script, params.prefix)


}