void test(struct A *a, s32 b) {
    D_410130 = (int *) a->array[b];
    D_410130 = &a->array[b];
    D_410130 = (s32) (a + (b * 8))->unk30;
    D_410130 = (void *) ((a + (b * 8)) + 0x30);
    D_410130 = (s32) a[b].y;
}
