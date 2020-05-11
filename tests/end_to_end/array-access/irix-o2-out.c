void test(struct A *a, s32 b) {
    void *temp_v1;

    D_410100 = (int *) a->array[b];
    D_410100 = (int *) &a->array[b];
    temp_v1 = a + (b * 8);
    D_410100 = (int *) temp_v1->unk30;
    D_410100 = (int *) (temp_v1 + 0x30);
    D_410100 = (int *) a[b].y;
}
