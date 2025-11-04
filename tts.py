from transformers import VitsModel, AutoTokenizer
import torch

model = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
# unable to directly read number
text = "1 2 3 4 5 6 7 8 9 10"
# but able to read this
text_2 = "một hai ba bốn năm sáu bảy tám chín mười"

text_3 = "Chào bạn, sau đây là lời giải chi tiết cho bài toán. Chúng ta cần tính giới hạn L bằng giới hạn khi ích tiến đến không của lốc-ne-pe của một cộng ích trên ích, tất cả mũ một trên ích. \
Bước một: Xác định dạng vô định. \
Trước hết, ta tìm giới hạn của cơ số: giới hạn khi ích tiến đến không của lốc-ne-pe của một cộng ích trên ích. \
Đây là dạng không trên không. Áp dụng quy tắc Lô-pi-tan, ta có: giới hạn khi ích tiến đến không của, phân số, một trên một cộng ích, trên một, bằng một. \
Tiếp theo, ta tìm giới hạn của số mũ: giới hạn khi ích tiến đến không của một trên ích, bằng vô cùng. \
Vậy, bài toán có dạng vô định một mũ vô cùng. \
Bước hai: Sử dụng phương pháp e mũ u. \
Ta viết lại L bằng e mũ của, giới hạn khi ích tiến đến không của, một trên ích, nhân với, lốc-ne-pe của, lốc-ne-pe của một cộng ích trên ích. \
Bước ba: Tính giới hạn của số mũ. \
Ta cần tập trung giải quyết giới hạn K bằng giới hạn khi ích tiến đến không của, phân số, lốc-ne-pe của lốc-ne-pe của một cộng ích trên ích, trên ích. \
Khi ích tiến đến không, biểu thức bên trong lốc-ne-pe tiến đến một. Do đó, tử số tiến đến lốc-ne-pe của một, bằng không. Mẫu số ích cũng tiến đến không. \
Đây tiếp tục là dạng không trên không. \
Bước bốn: Sử dụng khai triển Tay-lo.\
Ta sử dụng khai triển lốc-ne-pe của một cộng ích bằng ích trừ ích mũ hai trên hai, cộng với vô cùng bé bậc cao hơn.\
Thay vào cơ số: lốc-ne-pe của một cộng ích trên ích, bằng, một trừ ích trên hai, cộng với vô cùng bé bậc cao hơn.\
Bây giờ, thay cơ số này vào giới hạn K:\
K bằng giới hạn khi ích tiến đến không của, phân số, lốc-ne-pe của một trừ ích trên hai, trên ích.\
Ta lại sử dụng khai triển lốc-ne-pe của một cộng u xấp xỉ bằng u khi u tiến đến không.\
K bằng giới hạn khi ích tiến đến không của, phân số, trừ ích trên hai, trên ích.\
Kết quả K bằng trừ một trên hai.\
Bước năm: Kết luận.\
Ta đã tìm được giới hạn của số mũ là K bằng trừ một trên hai.\
Thay K trở lại, ta có L bằng e mũ K, bằng e mũ trừ một trên hai.\
Kết quả cuối cùng là L bằng một trên căn bậc hai của e."

inputs = tokenizer(text_3, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform
import scipy
import numpy as np


output_numpy = output.squeeze().cpu().numpy()

scipy.io.wavfile.write(f"number.wav", rate=model.config.sampling_rate, data=output_numpy)

print("Đã lưu file techno.wav thành công!")