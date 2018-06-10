from skimage.measure import compare_ssim, compare_psnr
import cv2

im1 = cv2.imread('./dataset/Test/Set5/butterfly_GT.bmp')
im2 = cv2.imread('bicubic.bmp')
im3 = cv2.imread('output.bmp')

ssim1 = compare_ssim(im1, im2, multichannel=True)
psnr1 = compare_psnr(im1, im2)

ssim2 = compare_ssim(im1, im3, multichannel=True)
psnr2 = compare_psnr(im1, im3)

print('Bicubic: SSIM:', ssim1, 'PSNR:', psnr1)
print('SRCNN: SSIM:', ssim2, 'PSNR:', psnr2)
