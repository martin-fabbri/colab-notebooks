using Images, ImageView, Statistics

function draw_seam(img, seam)
    img_w_seam = copy(img)
    width = size(img, 1)
    for i = 1: width
        img_w_seam[i, seam[i]] = RGB(1, 1, 1)
    end
    return img_w_seam
end