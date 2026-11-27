#include <cupti.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static FILE *trace;
static size_t dropped;

static void check(CUptiResult v) {
    if (v != CUPTI_SUCCESS) {
        const char *why = NULL;
        cuptiGetResultString(v, &why);
        fprintf(stderr,"CUPTI: %s\n",why ? why : "unknown error");
        exit(2);
    }
}

static void CUPTIAPI request(uint8_t **buffer,size_t *size,size_t *max_records) {
    *size=8u<<20;
    *max_records=0;
    if (posix_memalign((void **)buffer,64,*size)!=0) abort();
}

static void CUPTIAPI complete(CUcontext ctx,uint32_t stream,uint8_t *buffer,size_t size,size_t valid) {
    (void)size;
    CUpti_Activity *record = NULL;
    CUptiResult status;
    flockfile(trace);
    while ((status=cuptiActivityGetNextRecord(buffer,valid,&record))==CUPTI_SUCCESS) {
        if (record->kind!=CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) continue;
        const CUpti_ActivityKernel10 *k=(const CUpti_ActivityKernel10 *)record;
        fprintf(trace,"%" PRIu64 "\t%" PRIu64 "\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%u\t%d\t%d\t%" PRIu64 "\n",
                k->start,k->end,k->name,k->gridX,k->gridY,k->gridZ,k->blockX,k->blockY,k->blockZ,
                k->registersPerThread,k->staticSharedMemory,k->dynamicSharedMemory,k->graphNodeId);
    }
    if (status!=CUPTI_ERROR_MAX_LIMIT_REACHED) check(status);
    size_t count=0;
    check(cuptiActivityGetNumDroppedRecords(ctx,stream,&count));
    dropped+=count;
    funlockfile(trace);
    free(buffer);
}

void capture_init(void) {
    trace=fopen("kernels.tsv","wx");
    if (!trace) { perror("kernels.tsv");exit(2); }
    check(cuptiActivityRegisterCallbacks(request,complete));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
}
uint64_t gpu_clock(void) {
    uint64_t ns=0;
    check(cuptiGetTimestamp(&ns));
    return ns;
}
void capture_finish(void) {
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    check(cuptiActivityFlushAll(0));
    fprintf(stderr,"dropped=%zu\n",dropped);
    if (dropped!=0 || fflush(trace)!=0 || ferror(trace)) exit(3);
}
