Given ref_image, ref_mask, image

ref_feat = feature_extractor(ref_image)
feat = feature_extractor(image)
ref_mask = ref_mask.resize(feat.shape[1], feat.shape[2])
mask = zero_like(ref_mask)
for feat_pixel in feat:
    ref_feat_pixel = correspondence(feat_pixel, ref_feat)
    if ref_mask[ref_feat_pixel] == 1:
        mask[feat_pixel] = 1

output mask





Given coarse_mask, self_attention_maps

refine = True
when refine:
    mask = zero_like(coarse_mask)
    for attention_pixel in self_attention_maps:
        map = self_attention_maps[attention_pixel]
        pos_value = sum(map*coarse_mask)
        neg_value = sum(map*(1-coarse_mask))
        if pos_value > neg_value:
            mask[attention_pixel] = 1
    refine = False if mask == coarse_mask
    coarse_mask = mask

return coarse_mask



Given coarse_mask, self_attention_maps

mask = coarse_mask
do:
    coarse_mask = mask
    mask = zero_like(coarse_mask)
    for attention_pixel in self_attention_maps:
        map = self_attention_maps[attention_pixel]
        pos_value = map*coarse_mask
        neg_value = map*(1-coarse_mask)
        if sum(pos_value) > sum(neg_value):
            mask[attention_pixel] = 1
while coarse_mask != mask

return mask