void test(u32 a) {
    test_s8((s8) a);
    test_s16((s16) a);
    test_s32_div((s32) a);
    test_s32_mod((s32) a);
    test_u32_div(a);
    test_u32_mod(a);
}

void test_s8(s8 c) {
    s8 sp8;

    sp8 = c;
    foo((u32) ((s8) (u8) sp8 / 2));
    foo((s8) (u8) sp8 / 3);
    foo((s8) (u8) sp8 / 5);
    foo((s8) (u8) sp8 / 7);
    foo((s8) (u8) sp8 / 10);
    foo((s8) (u8) sp8 / 100);
    foo((s8) (u8) sp8 / 255);
    foo((s8) (u8) sp8 % 2);
    foo((s8) (u8) sp8 % 3);
    foo((s8) (u8) sp8 % 5);
    foo((s8) (u8) sp8 % 7);
    foo((s8) (u8) sp8 % 10);
    foo((s8) (u8) sp8 % 100);
    foo((s8) (u8) sp8 % 255);
}

void test_s16(s16 h) {
    s16 sp8;

    sp8 = h;
    foo((u32) (sp8 / 2));
    foo(sp8 / 3);
    foo(sp8 / 5);
    foo(sp8 / 7);
    foo(sp8 / 10);
    foo(sp8 / 100);
    foo(sp8 / 255);
    foo(sp8 / 360);
    foo(sp8 / 65534);
    foo(sp8 % 2);
    foo(sp8 % 3);
    foo(sp8 % 5);
    foo(sp8 % 7);
    foo(sp8 % 10);
    foo(sp8 % 100);
    foo(sp8 % 255);
    foo(sp8 % 360);
    foo(sp8 % 65534);
}

void test_s32_div(s32 d) {
    s32 sp8;
    s32 temp_r0_10;
    s32 temp_r0_11;
    s32 temp_r0_12;
    s32 temp_r0_8;
    s32 temp_r0_9;
    u32 temp_r0;
    u32 temp_r0_2;
    u32 temp_r0_3;
    u32 temp_r0_4;
    u32 temp_r0_5;
    u32 temp_r0_6;
    u32 temp_r0_7;

    sp8 = d;
    foo((u32) d);
    foo((u32) (sp8 / 2));
    foo(sp8 / 3);
    foo((u32) (sp8 / 4));
    foo(sp8 / 5);
    foo(sp8 / 6);
    foo(sp8 / 7);
    foo((u32) (sp8 / 8));
    foo(sp8 / 9);
    foo(sp8 / 10);
    foo(sp8 / 11);
    foo(sp8 / 12);
    foo(sp8 / 13);
    foo(sp8 / 14);
    foo(sp8 / 15);
    foo((u32) (sp8 / 16));
    foo(sp8 / 17);
    foo(sp8 / 18);
    foo(sp8 / 19);
    foo(sp8 / 20);
    foo(sp8 / 21);
    foo(sp8 / 22);
    foo(sp8 / 23);
    foo(sp8 / 24);
    foo(sp8 / 25);
    foo(sp8 / 26);
    foo(sp8 / 27);
    foo(sp8 / 28);
    foo(sp8 / 29);
    foo(sp8 / 30);
    foo(sp8 / 31);
    foo((u32) (sp8 / 32));
    foo(sp8 / 33);
    foo(sp8 / 100);
    foo(sp8 / 255);
    foo(sp8 / 360);
    foo(sp8 / 1000);
    foo(sp8 / 10000);
    foo(sp8 / 100000);
    foo(sp8 / 1000000);
    foo(sp8 / 10000000);
    foo(sp8 / 100000000);
    temp_r0 = sp8 / 1073741822;
    foo(temp_r0 + (temp_r0 >> 0x1FU));
    temp_r0_2 = sp8 / 1073741822;
    foo(temp_r0_2 + (temp_r0_2 >> 0x1FU));
    foo((u32) (sp8 / 1073741824));
    temp_r0_3 = sp8 / 1073741824;
    foo(temp_r0_3 + (temp_r0_3 >> 0x1FU));
    temp_r0_4 = sp8 / 2147483648;
    foo(temp_r0_4 + (temp_r0_4 >> 0x1FU));
    temp_r0_5 = sp8 / 2147483645;
    foo(temp_r0_5 + (temp_r0_5 >> 0x1FU));
    temp_r0_6 = sp8 / 2147483648;
    foo(temp_r0_6 + (temp_r0_6 >> 0x1FU));
    foo(sp8 / 2147483648);
    temp_r0_7 = sp8 / 715827883;
    foo(temp_r0_7 + (temp_r0_7 >> 0x1FU));
    temp_r0_8 = (s32) (MULT_HI(0x7FFFFFFD, sp8) - sp8) >> 0x1E;
    foo(temp_r0_8 + ((u32) temp_r0_8 >> 0x1FU));
    temp_r0_9 = (s32) MULT_HI(0x99999999, sp8) >> 2;
    foo(temp_r0_9 + ((u32) temp_r0_9 >> 0x1FU));
    temp_r0_10 = (s32) (MULT_HI(0x6DB6DB6D, sp8) - sp8) >> 2;
    foo(temp_r0_10 + ((u32) temp_r0_10 >> 0x1FU));
    temp_r0_11 = (s32) MULT_HI(0x99999999, sp8) >> 1;
    foo(temp_r0_11 + ((u32) temp_r0_11 >> 0x1FU));
    foo((u32) -(sp8 / 4));
    temp_r0_12 = (s32) ((sp8 / 3) - sp8) >> 1;
    foo(temp_r0_12 + ((u32) temp_r0_12 >> 0x1FU));
    foo((u32) -(sp8 / 2));
    foo((u32) (d / -65537));
}

void test_s32_mod(s32 d) {
    s32 sp8;
    s32 temp_r0_10;
    s32 temp_r0_6;
    s32 temp_r0_7;
    s32 temp_r0_8;
    s32 temp_r0_9;
    s32 temp_r3_3;
    u32 temp_r0;
    u32 temp_r0_11;
    u32 temp_r0_2;
    u32 temp_r0_3;
    u32 temp_r0_4;
    u32 temp_r0_5;
    u32 temp_r3;
    u32 temp_r3_2;

    sp8 = d;
    foo(0U);
    foo(sp8 % 2);
    foo(sp8 % 3);
    foo(sp8 % 4);
    foo(sp8 % 5);
    foo(sp8 % 6);
    foo(sp8 % 7);
    foo(sp8 % 8);
    foo(sp8 % 9);
    foo(sp8 % 10);
    foo(sp8 % 11);
    foo(sp8 % 12);
    foo(sp8 % 13);
    foo(sp8 % 14);
    foo(sp8 % 15);
    foo(sp8 % 16);
    foo(sp8 % 17);
    foo(sp8 % 18);
    foo(sp8 % 19);
    foo(sp8 % 20);
    foo(sp8 % 21);
    foo(sp8 % 22);
    foo(sp8 % 23);
    foo(sp8 % 24);
    foo(sp8 % 25);
    foo(sp8 % 26);
    foo(sp8 % 27);
    foo(sp8 % 28);
    foo(sp8 % 29);
    foo(sp8 % 30);
    foo(sp8 % 31);
    foo(sp8 - ((sp8 / 32) << 5));
    foo(sp8 % 33);
    foo(sp8 % 100);
    foo(sp8 % 255);
    foo(sp8 % 360);
    foo(sp8 % 1000);
    foo(sp8 % 10000);
    foo(sp8 % 100000);
    foo(sp8 % 1000000);
    foo(sp8 % 10000000);
    foo(sp8 % 100000000);
    temp_r0 = sp8 / 1073741822;
    foo(sp8 - ((temp_r0 + (temp_r0 >> 0x1FU)) * 0x3FFFFFFE));
    temp_r3 = sp8 / 1073741822;
    foo(sp8 - ((temp_r3 + (temp_r3 >> 0x1FU)) * 0x3FFFFFFF));
    foo(sp8 - ((sp8 / 1073741824) << 0x1E));
    temp_r0_2 = sp8 / 1073741824;
    foo(sp8 - ((temp_r0_2 + (temp_r0_2 >> 0x1FU)) * 0x40000001));
    temp_r0_3 = sp8 / 2147483648;
    foo(sp8 - ((temp_r0_3 + (temp_r0_3 >> 0x1FU)) * 0x7FFFFFFD));
    temp_r3_2 = sp8 / 2147483645;
    foo(sp8 - ((temp_r3_2 + (temp_r3_2 >> 0x1FU)) * 0x7FFFFFFE));
    temp_r0_4 = sp8 / 2147483648;
    foo(sp8 - ((temp_r0_4 + (temp_r0_4 >> 0x1FU)) * 0x7FFFFFFF));
    foo(d % 2147483648);
    temp_r0_5 = sp8 / 715827883;
    foo(sp8 - ((temp_r0_5 + (temp_r0_5 >> 0x1FU)) * 0x80000001));
    temp_r3_3 = (s32) (MULT_HI(0x7FFFFFFD, sp8) - sp8) >> 0x1E;
    foo(sp8 - ((temp_r3_3 + ((u32) temp_r3_3 >> 0x1FU)) * 0x80000002));
    temp_r0_6 = (s32) MULT_HI(0x99999999, sp8) >> 2;
    foo(sp8 - ((temp_r0_6 + ((u32) temp_r0_6 >> 0x1FU)) * -0xA));
    temp_r0_7 = (s32) (MULT_HI(0x6DB6DB6D, sp8) - sp8) >> 2;
    foo(sp8 - ((temp_r0_7 + ((u32) temp_r0_7 >> 0x1FU)) * -7));
    temp_r0_8 = (s32) MULT_HI(0x99999999, sp8) >> 1;
    foo(sp8 - ((temp_r0_8 + ((u32) temp_r0_8 >> 0x1FU)) * -5));
    temp_r0_9 = (s32) ((sp8 / 2) - sp8) >> 1;
    foo(sp8 - ((temp_r0_9 + ((u32) temp_r0_9 >> 0x1FU)) * -4));
    temp_r0_10 = (s32) ((sp8 / 3) - sp8) >> 1;
    foo(sp8 - ((temp_r0_10 + ((u32) temp_r0_10 >> 0x1FU)) * -3));
    temp_r0_11 = (sp8 / 2) - sp8;
    foo(sp8 - ((temp_r0_11 + (temp_r0_11 >> 0x1FU)) * -2));
    foo(sp8 % -65537);
}

void test_u32_div(u32 u) {
    u32 sp8;
    s32 temp_r3;
    s32 temp_r3_10;
    s32 temp_r3_11;
    s32 temp_r3_12;
    s32 temp_r3_13;
    s32 temp_r3_2;
    s32 temp_r3_3;
    s32 temp_r3_4;
    s32 temp_r3_5;
    s32 temp_r3_6;
    s32 temp_r3_7;
    s32 temp_r3_8;
    s32 temp_r3_9;

    sp8 = u;
    foo(u);
    foo(sp8 >> 1U);
    foo(sp8 / 3);
    foo(sp8 >> 2U);
    foo(sp8 / 5);
    foo(sp8 / 6);
    temp_r3 = sp8 / 7;
    foo((u32) (((u32) (sp8 - temp_r3) >> 1U) + temp_r3) >> 2U);
    foo(sp8 >> 3U);
    foo(sp8 / 9);
    foo(sp8 / 10);
    foo(sp8 / 11);
    foo(sp8 / 12);
    foo(sp8 / 13);
    temp_r3_2 = sp8 / 7;
    foo((u32) (((u32) (sp8 - temp_r3_2) >> 1U) + temp_r3_2) >> 3U);
    foo(sp8 / 15);
    foo(sp8 >> 4U);
    foo(sp8 / 17);
    foo(sp8 / 18);
    temp_r3_3 = MULTU_HI(0xAF286BCB, sp8);
    foo((u32) (((u32) (sp8 - temp_r3_3) >> 1U) + temp_r3_3) >> 4U);
    foo(sp8 / 20);
    temp_r3_4 = MULTU_HI(0x86186187, sp8);
    foo((u32) (((u32) (sp8 - temp_r3_4) >> 1U) + temp_r3_4) >> 4U);
    foo(sp8 / 22);
    foo(sp8 / 23);
    foo(sp8 / 24);
    foo(sp8 / 25);
    foo(sp8 / 26);
    temp_r3_5 = MULTU_HI(0x2F684BDB, sp8);
    foo((u32) (((u32) (sp8 - temp_r3_5) >> 1U) + temp_r3_5) >> 4U);
    temp_r3_6 = sp8 / 7;
    foo((u32) (((u32) (sp8 - temp_r3_6) >> 1U) + temp_r3_6) >> 4U);
    foo(sp8 / 29);
    foo(sp8 / 30);
    temp_r3_7 = sp8 / 31;
    foo((u32) (((u32) (sp8 - temp_r3_7) >> 1U) + temp_r3_7) >> 4U);
    foo(sp8 >> 5U);
    foo(sp8 / 33);
    foo(sp8 / 100);
    foo(sp8 / 255);
    temp_r3_8 = MULTU_HI(0x6C16C16D, sp8);
    foo((u32) (((u32) (sp8 - temp_r3_8) >> 1U) + temp_r3_8) >> 8U);
    foo(sp8 / 1000);
    foo(sp8 / 10000);
    temp_r3_9 = MULTU_HI(0x4F8B588F, sp8);
    foo((u32) (((u32) (sp8 - temp_r3_9) >> 1U) + temp_r3_9) >> 0x10U);
    foo(sp8 / 1000000);
    foo(sp8 / 10000000);
    foo(sp8 / 100000000);
    foo(sp8 >> 0x1EU);
    foo((u32) MULTU_HI(-0x10003, sp8) >> 0x1EU);
    temp_r3_10 = sp8 / 858993459;
    foo((u32) (((u32) (sp8 - temp_r3_10) >> 1U) + temp_r3_10) >> 0x1EU);
    temp_r3_11 = sp8 / 1431655765;
    foo((u32) (((u32) (sp8 - temp_r3_11) >> 1U) + temp_r3_11) >> 0x1EU);
    foo(sp8 / 2147483648);
    temp_r3_12 = MULTU_HI(-0x10003, sp8);
    foo((u32) (((u32) (sp8 - temp_r3_12) >> 1U) + temp_r3_12) >> 0x1FU);
    temp_r3_13 = sp8 / 1431655765;
    foo((u32) (((u32) (sp8 - temp_r3_13) >> 1U) + temp_r3_13) >> 0x1FU);
    foo((u32) (sp8 / 2) >> 0x1FU);
}

void test_u32_mod(u32 u) {
    u32 sp8;
    s32 temp_r3;
    s32 temp_r3_10;
    s32 temp_r3_2;
    s32 temp_r3_3;
    s32 temp_r3_4;
    s32 temp_r3_5;
    s32 temp_r3_6;
    s32 temp_r3_7;
    s32 temp_r3_8;
    s32 temp_r3_9;
    s32 temp_r4;
    s32 temp_r4_2;
    s32 temp_r4_3;

    sp8 = u;
    foo(0U);
    foo(sp8 & 1);
    foo(sp8 % 3);
    foo(sp8 & 3);
    foo(sp8 % 5);
    foo(u % 6);
    temp_r3 = sp8 / 7;
    foo(sp8 - (((u32) (((u32) (sp8 - temp_r3) >> 1U) + temp_r3) >> 2U) * 7));
    foo(sp8 & 7);
    foo(sp8 % 9);
    foo(u % 10);
    foo(sp8 % 11);
    foo(u % 12);
    foo(sp8 % 13);
    temp_r3_2 = sp8 / 7;
    foo(sp8 - (((u32) (((u32) (sp8 - temp_r3_2) >> 1U) + temp_r3_2) >> 3U) * 0xE));
    foo(sp8 % 15);
    foo(sp8 & 0xF);
    foo(sp8 % 17);
    foo(u % 18);
    temp_r3_3 = MULTU_HI(0xAF286BCB, sp8);
    foo(sp8 - (((u32) (((u32) (sp8 - temp_r3_3) >> 1U) + temp_r3_3) >> 4U) * 0x13));
    foo(u % 20);
    temp_r3_4 = MULTU_HI(0x86186187, sp8);
    foo(sp8 - (((u32) (((u32) (sp8 - temp_r3_4) >> 1U) + temp_r3_4) >> 4U) * 0x15));
    foo(u % 22);
    foo(sp8 % 23);
    foo(u % 24);
    foo(sp8 % 25);
    foo(u % 26);
    temp_r3_5 = MULTU_HI(0x2F684BDB, sp8);
    foo(sp8 - (((u32) (((u32) (sp8 - temp_r3_5) >> 1U) + temp_r3_5) >> 4U) * 0x1B));
    temp_r3_6 = sp8 / 7;
    foo(sp8 - (((u32) (((u32) (sp8 - temp_r3_6) >> 1U) + temp_r3_6) >> 4U) * 0x1C));
    foo(sp8 % 29);
    foo(u % 30);
    temp_r3_7 = sp8 / 31;
    foo(sp8 - (((u32) (((u32) (sp8 - temp_r3_7) >> 1U) + temp_r3_7) >> 4U) * 0x1F));
    foo(sp8 & 0x1F);
    foo(sp8 % 33);
    foo(u % 100);
    foo(sp8 % 255);
    temp_r3_8 = MULTU_HI(0x6C16C16D, sp8);
    foo(sp8 - (((u32) (((u32) (sp8 - temp_r3_8) >> 1U) + temp_r3_8) >> 8U) * 0x168));
    foo(sp8 % 1000);
    foo(sp8 % 10000);
    temp_r3_9 = MULTU_HI(0x4F8B588F, sp8);
    foo(sp8 - (((u32) (((u32) (sp8 - temp_r3_9) >> 1U) + temp_r3_9) >> 0x10U) * 0x186A0));
    foo(sp8 % 1000000);
    foo(sp8 % 10000000);
    foo(sp8 % 100000000);
    foo(sp8 & 0x3FFFFFFF);
    foo(sp8 - (((u32) MULTU_HI(-0x10003, sp8) >> 0x1EU) * 0x40000001));
    temp_r3_10 = sp8 / 858993459;
    foo(sp8 - (((u32) (((u32) (sp8 - temp_r3_10) >> 1U) + temp_r3_10) >> 0x1EU) * 0x7FFFFFFE));
    temp_r4 = sp8 / 1431655765;
    foo(sp8 - (((u32) (((u32) (sp8 - temp_r4) >> 1U) + temp_r4) >> 0x1EU) * 0x7FFFFFFF));
    foo(u % 2147483648);
    temp_r4_2 = MULTU_HI(-0x10003, sp8);
    foo(sp8 - (((u32) (((u32) (sp8 - temp_r4_2) >> 1U) + temp_r4_2) >> 0x1FU) * 0x80000001));
    temp_r4_3 = sp8 / 1431655765;
    foo(sp8 - (((u32) (((u32) (sp8 - temp_r4_3) >> 1U) + temp_r4_3) >> 0x1FU) * -0x10002));
    foo(sp8 % -65537U);
}
