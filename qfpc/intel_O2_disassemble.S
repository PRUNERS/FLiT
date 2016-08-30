Dump of assembler code for function Vector<double>::operator^(Vector<double> const&) const:
=> 0x0000000000415490 <+0>:	41 56	push   %r14
   0x0000000000415492 <+2>:	41 57	push   %r15
   0x0000000000415494 <+4>:	53	push   %rbx
   0x0000000000415495 <+5>:	48 83 ec 30	sub    $0x30,%rsp
   0x0000000000415499 <+9>:	ba d0 f7 43 00	mov    $0x43f7d0,%edx
   0x000000000041549e <+14>:	48 89 fb	mov    %rdi,%rbx
   0x00000000004154a1 <+17>:	48 8d 7c 24 18	lea    0x18(%rsp),%rdi
   0x00000000004154a6 <+22>:	48 89 7f f0	mov    %rdi,-0x10(%rdi)
   0x00000000004154aa <+26>:	49 89 f6	mov    %rsi,%r14
   0x00000000004154ad <+29>:	48 89 d1	mov    %rdx,%rcx
   0x00000000004154b0 <+32>:	48 83 e2 f0	and    $0xfffffffffffffff0,%rdx
   0x00000000004154b4 <+36>:	66 0f ef c0	pxor   %xmm0,%xmm0
   0x00000000004154b8 <+40>:	66 0f 74 02	pcmpeqb (%rdx),%xmm0
   0x00000000004154bc <+44>:	66 0f d7 c0	pmovmskb %xmm0,%eax
   0x00000000004154c0 <+48>:	83 e1 0f	and    $0xf,%ecx
   0x00000000004154c3 <+51>:	d3 e8	shr    %cl,%eax
   0x00000000004154c5 <+53>:	0f bc c0	bsf    %eax,%eax
   0x00000000004154c8 <+56>:	75 0b	jne    0x4154d5 <Vector<double>::operator^(Vector<double> const&) const+69>
   0x00000000004154ca <+58>:	48 89 d0	mov    %rdx,%rax
   0x00000000004154cd <+61>:	48 03 d1	add    %rcx,%rdx
   0x00000000004154d0 <+64>:	e8 db 84 01 00	callq  0x42d9b0 <__intel_sse2_strlen>
   0x00000000004154d5 <+69>:	49 89 c7	mov    %rax,%r15
   0x00000000004154d8 <+72>:	4c 89 7c 24 28	mov    %r15,0x28(%rsp)
   0x00000000004154dd <+77>:	49 83 ff 0f	cmp    $0xf,%r15
   0x00000000004154e1 <+81>:	77 0d	ja     0x4154f0 <Vector<double>::operator^(Vector<double> const&) const+96>
   0x00000000004154e3 <+83>:	49 83 ff 01	cmp    $0x1,%r15
   0x00000000004154e7 <+87>:	75 2a	jne    0x415513 <Vector<double>::operator^(Vector<double> const&) const+131>
   0x00000000004154e9 <+89>:	c6 44 24 18 6f	movb   $0x6f,0x18(%rsp)
   0x00000000004154ee <+94>:	eb 30	jmp    0x415520 <Vector<double>::operator^(Vector<double> const&) const+144>
   0x00000000004154f0 <+96>:	33 d2	xor    %edx,%edx
   0x00000000004154f2 <+98>:	48 8d 7c 24 08	lea    0x8(%rsp),%rdi
   0x00000000004154f7 <+103>:	48 8d 74 24 28	lea    0x28(%rsp),%rsi
   0x00000000004154fc <+108>:	e8 4f cd fe ff	callq  0x402250 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@plt>
   0x0000000000415501 <+113>:	48 89 c7	mov    %rax,%rdi
   0x0000000000415504 <+116>:	48 8b 44 24 28	mov    0x28(%rsp),%rax
   0x0000000000415509 <+121>:	48 89 7c 24 08	mov    %rdi,0x8(%rsp)
   0x000000000041550e <+126>:	48 89 44 24 18	mov    %rax,0x18(%rsp)
   0x0000000000415513 <+131>:	be d0 f7 43 00	mov    $0x43f7d0,%esi
   0x0000000000415518 <+136>:	4c 89 fa	mov    %r15,%rdx
   0x000000000041551b <+139>:	e8 10 83 01 00	callq  0x42d830 <_intel_fast_memcpy>
   0x0000000000415520 <+144>:	48 8b 44 24 28	mov    0x28(%rsp),%rax
   0x0000000000415525 <+149>:	48 8d 4c 24 18	lea    0x18(%rsp),%rcx
   0x000000000041552a <+154>:	48 8b 51 f0	mov    -0x10(%rcx),%rdx
   0x000000000041552e <+158>:	48 89 41 f8	mov    %rax,-0x8(%rcx)
   0x0000000000415532 <+162>:	c6 04 10 00	movb   $0x0,(%rax,%rdx,1)
   0x0000000000415536 <+166>:	48 8b 79 f0	mov    -0x10(%rcx),%rdi
   0x000000000041553a <+170>:	48 3b f9	cmp    %rcx,%rdi
   0x000000000041553d <+173>:	74 05	je     0x415544 <Vector<double>::operator^(Vector<double> const&) const+180>
   0x000000000041553f <+175>:	e8 fc cc fe ff	callq  0x402240 <_ZdlPv@plt>
   0x0000000000415544 <+180>:	83 7b 18 02	cmpl   $0x2,0x18(%rbx)
   0x0000000000415548 <+184>:	66 0f ef c0	pxor   %xmm0,%xmm0
   0x000000000041554c <+188>:	0f 84 44 04 00 00	je     0x415996 <Vector<double>::operator^(Vector<double> const&) const+1286>
   0x0000000000415552 <+194>:	48 8b 33	mov    (%rbx),%rsi
   0x0000000000415555 <+197>:	48 8b 5b 08	mov    0x8(%rbx),%rbx
   0x0000000000415559 <+201>:	48 2b de	sub    %rsi,%rbx
   0x000000000041555c <+204>:	48 c1 fb 03	sar    $0x3,%rbx
   0x0000000000415560 <+208>:	48 8b 3d e1 7e 23 00	mov    0x237ee1(%rip),%rdi        # 0x64d448 <_ZN7GlobalsIdE5prodsE>
   0x0000000000415567 <+215>:	48 85 db	test   %rbx,%rbx #rbx is for(i, size() ; rdi is &i
   0x000000000041556a <+218>:	4c 8b 05 df 7e 23 00	mov    0x237edf(%rip),%r8        # 0x64d450 <_ZN7GlobalsIdE5prodsE+8> 
   0x0000000000415571 <+225>:	0f 86 11 03 00 00	jbe    0x415888 <Vector<double>::operator^(Vector<double> const&) const+1016> #jump if rbx == 0, testing for zero length 
   0x0000000000415577 <+231>:	49 8b 16	mov    (%r14),%rdx
   0x000000000041557a <+234>:	48 83 fb 06	cmp    $0x6,%rbx
   0x000000000041557e <+238>:	0f 8e 8d 02 00 00	jle    0x415811 <Vector<double>::operator^(Vector<double> const&) const+897> #why are we comparing size to 0x6?
   0x0000000000415584 <+244>:	48 3b fa	cmp    %rdx,%rdi
   0x0000000000415587 <+247>:	76 13	jbe    0x41559c <Vector<double>::operator^(Vector<double> const&) const+268>
   0x0000000000415589 <+249>:	48 89 f8	mov    %rdi,%rax
   0x000000000041558c <+252>:	48 8d 0c dd 00 00 00 00	lea    0x0(,%rbx,8),%rcx
   0x0000000000415594 <+260>:	48 2b c2	sub    %rdx,%rax
   0x0000000000415597 <+263>:	48 3b c8	cmp    %rax,%rcx
   0x000000000041559a <+266>:	7e 20	jle    0x4155bc <Vector<double>::operator^(Vector<double> const&) const+300>
   0x000000000041559c <+268>:	48 3b d7	cmp    %rdi,%rdx
   0x000000000041559f <+271>:	0f 86 6c 02 00 00	jbe    0x415811 <Vector<double>::operator^(Vector<double> const&) const+897>
   0x00000000004155a5 <+277>:	48 89 d0	mov    %rdx,%rax
	0x00000000004155a8 <+280>:	48 8d 0c dd 00 00 00 00	lea    0x0(,%rbx,8),%rcx
	#rcx is the extent of the loop, i.e. sizeof(T) * size()
   0x00000000004155b0 <+288>:	48 2b c7	sub    %rdi,%rax
   0x00000000004155b3 <+291>:	48 3b c1	cmp    %rcx,%rax
   0x00000000004155b6 <+294>:	0f 8c 55 02 00 00	jl     0x415811 <Vector<double>::operator^(Vector<double> const&) const+897>
   0x00000000004155bc <+300>:	48 3b fe	cmp    %rsi,%rdi
   0x00000000004155bf <+303>:	76 0b	jbe    0x4155cc <Vector<double>::operator^(Vector<double> const&) const+316>
   0x00000000004155c1 <+305>:	48 89 f8	mov    %rdi,%rax
   0x00000000004155c4 <+308>:	48 2b c6	sub    %rsi,%rax
   0x00000000004155c7 <+311>:	48 3b c8	cmp    %rax,%rcx
   0x00000000004155ca <+314>:	7e 18	jle    0x4155e4 <Vector<double>::operator^(Vector<double> const&) const+340>
   0x00000000004155cc <+316>:	48 3b f7	cmp    %rdi,%rsi
   0x00000000004155cf <+319>:	0f 86 3c 02 00 00	jbe    0x415811 <Vector<double>::operator^(Vector<double> const&) const+897>
   0x00000000004155d5 <+325>:	48 89 f0	mov    %rsi,%rax
   0x00000000004155d8 <+328>:	48 2b c7	sub    %rdi,%rax
   0x00000000004155db <+331>:	48 3b c1	cmp    %rcx,%rax
   0x00000000004155de <+334>:	0f 8c 2d 02 00 00	jl     0x415811 <Vector<double>::operator^(Vector<double> const&) const+897>
   0x00000000004155e4 <+340>:	45 32 c9	xor    %r9b,%r9b
   0x00000000004155e7 <+343>:	48 83 fb 08	cmp    $0x8,%rbx
   0x00000000004155eb <+347>:	0f 8c af 03 00 00	jl     0x4159a0 <Vector<double>::operator^(Vector<double> const&) const+1296>
   0x00000000004155f1 <+353>:	48 89 f9	mov    %rdi,%rcx
   0x00000000004155f4 <+356>:	48 83 e1 0f	and    $0xf,%rcx
   0x00000000004155f8 <+360>:	74 12	je     0x41560c <Vector<double>::operator^(Vector<double> const&) const+380>
   0x00000000004155fa <+362>:	48 f7 c1 07 00 00 00	test   $0x7,%rcx
   0x0000000000415601 <+369>:	0f 85 99 03 00 00	jne    0x4159a0 <Vector<double>::operator^(Vector<double> const&) const+1296>
   0x0000000000415607 <+375>:	b9 01 00 00 00	mov    $0x1,%ecx
   0x000000000041560c <+380>:	48 8d 41 08	lea    0x8(%rcx),%rax
   0x0000000000415610 <+384>:	48 3b d8	cmp    %rax,%rbx
   0x0000000000415613 <+387>:	0f 8c 87 03 00 00	jl     0x4159a0 <Vector<double>::operator^(Vector<double> const&) const+1296>
   0x0000000000415619 <+393>:	48 89 d8	mov    %rbx,%rax
   0x000000000041561c <+396>:	45 33 db	xor    %r11d,%r11d
   0x000000000041561f <+399>:	48 2b c1	sub    %rcx,%rax
   0x0000000000415622 <+402>:	45 33 d2	xor    %r10d,%r10d
   0x0000000000415625 <+405>:	48 83 e0 07	and    $0x7,%rax
   0x0000000000415629 <+409>:	48 f7 d8	neg    %rax
   0x000000000041562c <+412>:	48 03 c3	add    %rbx,%rax
   0x000000000041562f <+415>:	48 85 c9	test   %rcx,%rcx
   0x0000000000415632 <+418>:	76 1d	jbe    0x415651 <Vector<double>::operator^(Vector<double> const&) const+449>
   0x0000000000415634 <+420>:	41 ff c3	inc    %r11d
   0x0000000000415637 <+423>:	f2 42 0f 10 0c d6	movsd  (%rsi,%r10,8),%xmm1
   0x000000000041563d <+429>:	f2 42 0f 59 0c d2	mulsd  (%rdx,%r10,8),%xmm1
   0x0000000000415643 <+435>:	f2 42 0f 11 0c d7	movsd  %xmm1,(%rdi,%r10,8)
   0x0000000000415649 <+441>:	49 ff c2	inc    %r10
   0x000000000041564c <+444>:	4c 3b d9	cmp    %rcx,%r11
   0x000000000041564f <+447>:	72 e3	jb     0x415634 <Vector<double>::operator^(Vector<double> const&) const+420>
   0x0000000000415651 <+449>:	41 89 ca	mov    %ecx,%r10d 
   0x0000000000415654 <+452>:	4c 8d 1c ca	lea    (%rdx,%rcx,8),%r11 #rdx is pointing to data[0] or rhs[0]
   0x0000000000415658 <+456>:	49 f7 c3 0f 00 00 00	test   $0xf,%r11 #alignment check?
   0x000000000041565f <+463>:	0f 84 9b 00 00 00	je     0x415700 <Vector<double>::operator^(Vector<double> const&) const+624>
   0x0000000000415665 <+469>:	41 83 c2 08	add    $0x8,%r10d
   0x0000000000415669 <+473>:	f2 0f 10 14 ce	movsd  (%rsi,%rcx,8),%xmm2
   0x000000000041566e <+478>:	f2 0f 10 64 ce 10	movsd  0x10(%rsi,%rcx,8),%xmm4
   0x0000000000415674 <+484>:	f2 0f 10 74 ce 20	movsd  0x20(%rsi,%rcx,8),%xmm6
   0x000000000041567a <+490>:	f2 44 0f 10 44 ce 30	movsd  0x30(%rsi,%rcx,8),%xmm8
   0x0000000000415681 <+497>:	f2 0f 10 0c ca	movsd  (%rdx,%rcx,8),%xmm1
   0x0000000000415686 <+502>:	f2 0f 10 5c ca 10	movsd  0x10(%rdx,%rcx,8),%xmm3
   0x000000000041568c <+508>:	f2 0f 10 6c ca 20	movsd  0x20(%rdx,%rcx,8),%xmm5
   0x0000000000415692 <+514>:	f2 0f 10 7c ca 30	movsd  0x30(%rdx,%rcx,8),%xmm7
   0x0000000000415698 <+520>:	66 0f 16 4c ca 08	movhpd 0x8(%rdx,%rcx,8),%xmm1
   0x000000000041569e <+526>:	66 0f 16 5c ca 18	movhpd 0x18(%rdx,%rcx,8),%xmm3
   0x00000000004156a4 <+532>:	66 0f 16 6c ca 28	movhpd 0x28(%rdx,%rcx,8),%xmm5
   0x00000000004156aa <+538>:	66 0f 16 7c ca 38	movhpd 0x38(%rdx,%rcx,8),%xmm7
   0x00000000004156b0 <+544>:	66 0f 16 54 ce 08	movhpd 0x8(%rsi,%rcx,8),%xmm2
   0x00000000004156b6 <+550>:	66 0f 16 64 ce 18	movhpd 0x18(%rsi,%rcx,8),%xmm4
   0x00000000004156bc <+556>:	66 0f 16 74 ce 28	movhpd 0x28(%rsi,%rcx,8),%xmm6
   0x00000000004156c2 <+562>:	66 44 0f 16 44 ce 38	movhpd 0x38(%rsi,%rcx,8),%xmm8
   0x00000000004156c9 <+569>:	66 0f 59 d1	mulpd  %xmm1,%xmm2
   0x00000000004156cd <+573>:	66 0f 59 e3	mulpd  %xmm3,%xmm4
   0x00000000004156d1 <+577>:	66 0f 59 f5	mulpd  %xmm5,%xmm6
   0x00000000004156d5 <+581>:	66 44 0f 59 c7	mulpd  %xmm7,%xmm8
   0x00000000004156da <+586>:	0f 29 14 cf	movaps %xmm2,(%rdi,%rcx,8)
   0x00000000004156de <+590>:	0f 29 64 cf 10	movaps %xmm4,0x10(%rdi,%rcx,8)
   0x00000000004156e3 <+595>:	0f 29 74 cf 20	movaps %xmm6,0x20(%rdi,%rcx,8)
   0x00000000004156e8 <+600>:	44 0f 29 44 cf 30	movaps %xmm8,0x30(%rdi,%rcx,8)
   0x00000000004156ee <+606>:	48 83 c1 08	add    $0x8,%rcx
   0x00000000004156f2 <+610>:	4c 3b d0	cmp    %rax,%r10
   0x00000000004156f5 <+613>:	0f 82 6a ff ff ff	jb     0x415665 <Vector<double>::operator^(Vector<double> const&) const+469>
   0x00000000004156fb <+619>:	eb 69	jmp    0x415766 <Vector<double>::operator^(Vector<double> const&) const+726>
   0x00000000004156fd <+621>:	0f 1f 00	nopl   (%rax)
   0x0000000000415700 <+624>:	41 83 c2 08	add    $0x8,%r10d
   0x0000000000415704 <+628>:	f2 0f 10 0c ce	movsd  (%rsi,%rcx,8),%xmm1
   0x0000000000415709 <+633>:	f2 0f 10 54 ce 10	movsd  0x10(%rsi,%rcx,8),%xmm2
   0x000000000041570f <+639>:	f2 0f 10 5c ce 20	movsd  0x20(%rsi,%rcx,8),%xmm3
   0x0000000000415715 <+645>:	f2 0f 10 64 ce 30	movsd  0x30(%rsi,%rcx,8),%xmm4
   0x000000000041571b <+651>:	66 0f 16 4c ce 08	movhpd 0x8(%rsi,%rcx,8),%xmm1
   0x0000000000415721 <+657>:	66 0f 16 54 ce 18	movhpd 0x18(%rsi,%rcx,8),%xmm2
   0x0000000000415727 <+663>:	66 0f 16 5c ce 28	movhpd 0x28(%rsi,%rcx,8),%xmm3
   0x000000000041572d <+669>:	66 0f 16 64 ce 38	movhpd 0x38(%rsi,%rcx,8),%xmm4
   0x0000000000415733 <+675>:	66 0f 59 0c ca	mulpd  (%rdx,%rcx,8),%xmm1
   0x0000000000415738 <+680>:	66 0f 59 54 ca 10	mulpd  0x10(%rdx,%rcx,8),%xmm2
   0x000000000041573e <+686>:	66 0f 59 5c ca 20	mulpd  0x20(%rdx,%rcx,8),%xmm3
   0x0000000000415744 <+692>:	66 0f 59 64 ca 30	mulpd  0x30(%rdx,%rcx,8),%xmm4
   0x000000000041574a <+698>:	0f 29 0c cf	movaps %xmm1,(%rdi,%rcx,8)
   0x000000000041574e <+702>:	0f 29 54 cf 10	movaps %xmm2,0x10(%rdi,%rcx,8)
   0x0000000000415753 <+707>:	0f 29 5c cf 20	movaps %xmm3,0x20(%rdi,%rcx,8)
   0x0000000000415758 <+712>:	0f 29 64 cf 30	movaps %xmm4,0x30(%rdi,%rcx,8)
   0x000000000041575d <+717>:	48 83 c1 08	add    $0x8,%rcx
   0x0000000000415761 <+721>:	4c 3b d0	cmp    %rax,%r10
   0x0000000000415764 <+724>:	72 9a	jb     0x415700 <Vector<double>::operator^(Vector<double> const&) const+624>
   0x0000000000415766 <+726>:	48 8d 48 01	lea    0x1(%rax),%rcx
   0x000000000041576a <+730>:	48 3b d9	cmp    %rcx,%rbx
   0x000000000041576d <+733>:	0f 82 15 01 00 00	jb     0x415888 <Vector<double>::operator^(Vector<double> const&) const+1016>
   0x0000000000415773 <+739>:	48 2b d8	sub    %rax,%rbx
   0x0000000000415776 <+742>:	41 80 f9 01	cmp    $0x1,%r9b
   0x000000000041577a <+746>:	75 04	jne    0x415780 <Vector<double>::operator^(Vector<double> const&) const+752>
   0x000000000041577c <+748>:	33 c9	xor    %ecx,%ecx
   0x000000000041577e <+750>:	eb 52	jmp    0x4157d2 <Vector<double>::operator^(Vector<double> const&) const+834>
   0x0000000000415780 <+752>:	48 83 fb 02	cmp    $0x2,%rbx
   0x0000000000415784 <+756>:	7c f6	jl     0x41577c <Vector<double>::operator^(Vector<double> const&) const+748>
   0x0000000000415786 <+758>:	4c 63 d0	movslq %eax,%r10
   0x0000000000415789 <+761>:	48 89 d9	mov    %rbx,%rcx
   0x000000000041578c <+764>:	48 83 e1 fe	and    $0xfffffffffffffffe,%rcx
   0x0000000000415790 <+768>:	45 33 c9	xor    %r9d,%r9d
   0x0000000000415793 <+771>:	4e 8d 1c d7	lea    (%rdi,%r10,8),%r11
   0x0000000000415797 <+775>:	4e 8d 34 d6	lea    (%rsi,%r10,8),%r14
   0x000000000041579b <+779>:	4e 8d 3c d2	lea    (%rdx,%r10,8),%r15
   0x000000000041579f <+783>:	45 33 d2	xor    %r10d,%r10d
   0x00000000004157a2 <+786>:	41 83 c1 02	add    $0x2,%r9d
   0x00000000004157a6 <+790>:	f2 43 0f 10 14 d6	movsd  (%r14,%r10,8),%xmm2
   0x00000000004157ac <+796>:	f2 43 0f 10 0c d7	movsd  (%r15,%r10,8),%xmm1
   0x00000000004157b2 <+802>:	66 43 0f 16 4c d7 08	movhpd 0x8(%r15,%r10,8),%xmm1
   0x00000000004157b9 <+809>:	66 43 0f 16 54 d6 08	movhpd 0x8(%r14,%r10,8),%xmm2
   0x00000000004157c0 <+816>:	66 0f 59 d1	mulpd  %xmm1,%xmm2
   0x00000000004157c4 <+820>:	43 0f 29 14 d3	movaps %xmm2,(%r11,%r10,8)
   0x00000000004157c9 <+825>:	49 83 c2 02	add    $0x2,%r10
   0x00000000004157cd <+829>:	4c 3b c9	cmp    %rcx,%r9
   0x00000000004157d0 <+832>:	72 d0	jb     0x4157a2 <Vector<double>::operator^(Vector<double> const&) const+786>
   0x00000000004157d2 <+834>:	4c 63 d1	movslq %ecx,%r10
   0x00000000004157d5 <+837>:	41 89 c9	mov    %ecx,%r9d
   0x00000000004157d8 <+840>:	89 c9	mov    %ecx,%ecx
   0x00000000004157da <+842>:	48 3b cb	cmp    %rbx,%rcx
   0x00000000004157dd <+845>:	0f 83 a5 00 00 00	jae    0x415888 <Vector<double>::operator^(Vector<double> const&) const+1016>
   0x00000000004157e3 <+851>:	48 63 c0	movslq %eax,%rax
   0x00000000004157e6 <+854>:	4c 8d 1c c7	lea    (%rdi,%rax,8),%r11
   0x00000000004157ea <+858>:	48 8d 0c c6	lea    (%rsi,%rax,8),%rcx
   0x00000000004157ee <+862>:	48 8d 04 c2	lea    (%rdx,%rax,8),%rax
   0x00000000004157f2 <+866>:	41 ff c1	inc    %r9d
   0x00000000004157f5 <+869>:	f2 42 0f 10 0c d1	movsd  (%rcx,%r10,8),%xmm1
   0x00000000004157fb <+875>:	f2 42 0f 59 0c d0	mulsd  (%rax,%r10,8),%xmm1
   0x0000000000415801 <+881>:	f2 43 0f 11 0c d3	movsd  %xmm1,(%r11,%r10,8)
   0x0000000000415807 <+887>:	49 ff c2	inc    %r10
   0x000000000041580a <+890>:	4c 3b cb	cmp    %rbx,%r9
   0x000000000041580d <+893>:	72 e3	jb     0x4157f2 <Vector<double>::operator^(Vector<double> const&) const+866>
   0x000000000041580f <+895>:	eb 77	jmp    0x415888 <Vector<double>::operator^(Vector<double> const&) const+1016>
   0x0000000000415811 <+897>:	48 89 d8	mov    %rbx,%rax
   0x0000000000415814 <+900>:	41 b9 01 00 00 00	mov    $0x1,%r9d
   0x000000000041581a <+906>:	48 c1 e8 3f	shr    $0x3f,%rax
   0x000000000041581e <+910>:	33 c9	xor    %ecx,%ecx
   0x0000000000415820 <+912>:	48 03 c3	add    %rbx,%rax
   0x0000000000415823 <+915>:	48 d1 f8	sar    %rax
   0x0000000000415826 <+918>:	48 85 c0	test   %rax,%rax
   0x0000000000415829 <+921>:	76 40	jbe    0x41586b <Vector<double>::operator^(Vector<double> const&) const+987>
   0x000000000041582b <+923>:	44 8d 0c 09	lea    (%rcx,%rcx,1),%r9d
   0x000000000041582f <+927>:	4d 63 c9	movslq %r9d,%r9
   0x0000000000415832 <+930>:	ff c1	inc    %ecx
   0x0000000000415834 <+932>:	48 3b c8	cmp    %rax,%rcx
   0x0000000000415837 <+935>:	f2 42 0f 10 0c ce	movsd  (%rsi,%r9,8),%xmm1
   0x000000000041583d <+941>:	f2 42 0f 59 0c ca	mulsd  (%rdx,%r9,8),%xmm1
   0x0000000000415843 <+947>:	f2 42 0f 11 0c cf	movsd  %xmm1,(%rdi,%r9,8)
   0x0000000000415849 <+953>:	f2 42 0f 10 54 ce 08	movsd  0x8(%rsi,%r9,8),%xmm2
   0x0000000000415850 <+960>:	f2 42 0f 59 54 ca 08	mulsd  0x8(%rdx,%r9,8),%xmm2
   0x0000000000415857 <+967>:	f2 42 0f 11 54 cf 08	movsd  %xmm2,0x8(%rdi,%r9,8)
   0x000000000041585e <+974>:	72 cb	jb     0x41582b <Vector<double>::operator^(Vector<double> const&) const+923>
   0x0000000000415860 <+976>:	48 63 c9	movslq %ecx,%rcx
   0x0000000000415863 <+979>:	4c 8d 0c 4d 01 00 00 00	lea    0x1(,%rcx,2),%r9
   0x000000000041586b <+987>:	49 ff c9	dec    %r9
   0x000000000041586e <+990>:	44 89 c8	mov    %r9d,%eax
   0x0000000000415871 <+993>:	48 3b c3	cmp    %rbx,%rax
   0x0000000000415874 <+996>:	73 12	jae    0x415888 <Vector<double>::operator^(Vector<double> const&) const+1016>
   0x0000000000415876 <+998>:	49 63 c1	movslq %r9d,%rax
   0x0000000000415879 <+1001>:	f2 0f 10 0c c6	movsd  (%rsi,%rax,8),%xmm1
   0x000000000041587e <+1006>:	f2 0f 59 0c c2	mulsd  (%rdx,%rax,8),%xmm1
   0x0000000000415883 <+1011>:	f2 0f 11 0c c7	movsd  %xmm1,(%rdi,%rax,8)
   0x0000000000415888 <+1016>:	49 3b f8	cmp    %r8,%rdi
   0x000000000041588b <+1019>:	0f 84 05 01 00 00	je     0x415996 <Vector<double>::operator^(Vector<double> const&) const+1286>
   0x0000000000415891 <+1025>:	4c 89 c0	mov    %r8,%rax
   0x0000000000415894 <+1028>:	48 2b c7	sub    %rdi,%rax #rdi is prods.data()
   0x0000000000415897 <+1031>:	48 83 c0 07	add    $0x7,%rax
   0x000000000041589b <+1035>:	48 c1 f8 02	sar    $0x2,%rax
   0x000000000041589f <+1039>:	48 c1 e8 3d	shr    $0x3d,%rax
   0x00000000004158a3 <+1043>:	48 2b c7	sub    %rdi,%rax
   0x00000000004158a6 <+1046>:	4a 8d 5c 00 07	lea    0x7(%rax,%r8,1),%rbx
   0x00000000004158ab <+1051>:	48 c1 fb 03	sar    $0x3,%rbx
   0x00000000004158af <+1055>:	48 83 fb 10	cmp    $0x10,%rbx
   0x00000000004158b3 <+1059>:	0f 8c f1 00 00 00	jl     0x4159aa <Vector<double>::operator^(Vector<double> const&) const+1306>
   0x00000000004158b9 <+1065>:	48 89 fa	mov    %rdi,%rdx
   0x00000000004158bc <+1068>:	48 83 e2 0f	and    $0xf,%rdx #alignment check
   0x00000000004158c0 <+1072>:	74 12	je     0x4158d4 <Vector<double>::operator^(Vector<double> const&) const+1092>
   0x00000000004158c2 <+1074>:	48 f7 c2 07 00 00 00	test   $0x7,%rdx
   0x00000000004158c9 <+1081>:	0f 85 db 00 00 00	jne    0x4159aa <Vector<double>::operator^(Vector<double> const&) const+1306>
   0x00000000004158cf <+1087>:	ba 01 00 00 00	mov    $0x1,%edx
   0x00000000004158d4 <+1092>:	48 8d 42 10	lea    0x10(%rdx),%rax
   0x00000000004158d8 <+1096>:	48 3b d8	cmp    %rax,%rbx
   0x00000000004158db <+1099>:	0f 8c c9 00 00 00	jl     0x4159aa <Vector<double>::operator^(Vector<double> const&) const+1306>
   0x00000000004158e1 <+1105>:	48 89 d9	mov    %rbx,%rcx
   0x00000000004158e4 <+1108>:	33 c0	xor    %eax,%eax
   0x00000000004158e6 <+1110>:	48 2b ca	sub    %rdx,%rcx
   0x00000000004158e9 <+1113>:	48 83 e1 0f	and    $0xf,%rcx
   0x00000000004158ed <+1117>:	48 f7 d9	neg    %rcx
   0x00000000004158f0 <+1120>:	48 03 cb	add    %rbx,%rcx
   0x00000000004158f3 <+1123>:	48 85 d2	test   %rdx,%rdx
   0x00000000004158f6 <+1126>:	76 0d	jbe    0x415905 <Vector<double>::operator^(Vector<double> const&) const+1141>
   0x00000000004158f8 <+1128>:	f2 0f 58 04 c7	addsd  (%rdi,%rax,8),%xmm0
   0x00000000004158fd <+1133>:	48 ff c0	inc    %rax
   0x0000000000415900 <+1136>:	48 3b c2	cmp    %rdx,%rax
   0x0000000000415903 <+1139>:	72 f3	jb     0x4158f8 <Vector<double>::operator^(Vector<double> const&) const+1128>
   0x0000000000415905 <+1141>:	66 0f ef f6	pxor   %xmm6,%xmm6 #we're zeroing out the sse registers here
   0x0000000000415909 <+1145>:	0f 28 fe	movaps %xmm6,%xmm7
   0x000000000041590c <+1148>:	0f 28 ee	movaps %xmm6,%xmm5
   0x000000000041590f <+1151>:	f2 0f 10 f8	movsd  %xmm0,%xmm7
   0x0000000000415913 <+1155>:	0f 28 e6	movaps %xmm6,%xmm4
   0x0000000000415916 <+1158>:	0f 28 de	movaps %xmm6,%xmm3
   0x0000000000415919 <+1161>:	0f 28 d6	movaps %xmm6,%xmm2
   0x000000000041591c <+1164>:	0f 28 ce	movaps %xmm6,%xmm1
   0x000000000041591f <+1167>:	0f 28 c6	movaps %xmm6,%xmm0
   0x0000000000415922 <+1170>:	66 0f 58 3c d7	addpd  (%rdi,%rdx,8),%xmm7
   0x0000000000415927 <+1175>:	66 0f 58 74 d7 10	addpd  0x10(%rdi,%rdx,8),%xmm6
   0x000000000041592d <+1181>:	66 0f 58 6c d7 20	addpd  0x20(%rdi,%rdx,8),%xmm5
   0x0000000000415933 <+1187>:	66 0f 58 64 d7 30	addpd  0x30(%rdi,%rdx,8),%xmm4
   0x0000000000415939 <+1193>:	66 0f 58 5c d7 40	addpd  0x40(%rdi,%rdx,8),%xmm3
   0x000000000041593f <+1199>:	66 0f 58 54 d7 50	addpd  0x50(%rdi,%rdx,8),%xmm2
   0x0000000000415945 <+1205>:	66 0f 58 4c d7 60	addpd  0x60(%rdi,%rdx,8),%xmm1
   0x000000000041594b <+1211>:	66 0f 58 44 d7 70	addpd  0x70(%rdi,%rdx,8),%xmm0
   0x0000000000415951 <+1217>:	48 83 c2 10	add    $0x10,%rdx
   0x0000000000415955 <+1221>:	48 3b d1	cmp    %rcx,%rdx
   0x0000000000415958 <+1224>:	72 c8	jb     0x415922 <Vector<double>::operator^(Vector<double> const&) const+1170>
   0x000000000041595a <+1226>:	66 0f 58 fe	addpd  %xmm6,%xmm7
   0x000000000041595e <+1230>:	66 0f 58 ec	addpd  %xmm4,%xmm5
   0x0000000000415962 <+1234>:	66 0f 58 da	addpd  %xmm2,%xmm3
   0x0000000000415966 <+1238>:	66 0f 58 c8	addpd  %xmm0,%xmm1
   0x000000000041596a <+1242>:	66 0f 58 fd	addpd  %xmm5,%xmm7
   0x000000000041596e <+1246>:	66 0f 58 d9	addpd  %xmm1,%xmm3
   0x0000000000415972 <+1250>:	66 0f 58 fb	addpd  %xmm3,%xmm7
   0x0000000000415976 <+1254>:	0f 28 c7	movaps %xmm7,%xmm0
   0x0000000000415979 <+1257>:	66 0f 15 c7	unpckhpd %xmm7,%xmm0
   0x000000000041597d <+1261>:	f2 0f 58 f8	addsd  %xmm0,%xmm7
   0x0000000000415981 <+1265>:	0f 28 c7	movaps %xmm7,%xmm0
   0x0000000000415984 <+1268>:	48 3b cb	cmp    %rbx,%rcx
   0x0000000000415987 <+1271>:	73 0d	jae    0x415996 <Vector<double>::operator^(Vector<double> const&) const+1286>
   0x0000000000415989 <+1273>:	f2 0f 58 04 cf	addsd  (%rdi,%rcx,8),%xmm0 #this section is where the oddballs, i.e. outside of 8x8 blocks is accumulated
   0x000000000041598e <+1278>:	48 ff c1	inc    %rcx
   0x0000000000415991 <+1281>:	48 3b cb	cmp    %rbx,%rcx
   0x0000000000415994 <+1284>:	72 f3	jb     0x415989 <Vector<double>::operator^(Vector<double> const&) const+1273>
   0x0000000000415996 <+1286>:	48 83 c4 30	add    $0x30,%rsp
   0x000000000041599a <+1290>:	5b	pop    %rbx
   0x000000000041599b <+1291>:	41 5f	pop    %r15
   0x000000000041599d <+1293>:	41 5e	pop    %r14
   0x000000000041599f <+1295>:	c3	retq   
   0x00000000004159a0 <+1296>:	41 b1 01	mov    $0x1,%r9b
   0x00000000004159a3 <+1299>:	33 c0	xor    %eax,%eax
   0x00000000004159a5 <+1301>:	e9 bc fd ff ff	jmpq   0x415766 <Vector<double>::operator^(Vector<double> const&) const+726>
   0x00000000004159aa <+1306>:	33 c9	xor    %ecx,%ecx
   0x00000000004159ac <+1308>:	eb d6	jmp    0x415984 <Vector<double>::operator^(Vector<double> const&) const+1268>
   0x00000000004159ae <+1310>:	48 89 04 24	mov    %rax,(%rsp)
   0x00000000004159b2 <+1314>:	48 8b 3c 24	mov    (%rsp),%rdi
   0x00000000004159b6 <+1318>:	e8 95 cb fe ff	callq  0x402550 <_Unwind_Resume@plt>
   0x00000000004159bb <+1323>:	0f 1f 44 00 00	nopl   0x0(%rax,%rax,1)
End of assembler dump.
