
fps = 8.0
frame_cnt = 400
ratio = 10.0 / fps

if frame_cnt > 60.0 * fps:
    frame_cnt = int(60.0 * fps)

if ratio > 1.0:
    frame_cnt = int(frame_cnt * ratio)

temps = [i for i in range(frame_cnt)]
frames = []

cnt = 0
if ratio > 1.0:
    for i, temp in enumerate(temps):
        if cnt < int(i * ratio):
            for _ in range(cnt, int(i * ratio)):
                frames.append(temps[i - 1])
                cnt += 1

                if cnt == frame_cnt:
                    break

        if cnt == frame_cnt:
            break

        frames.append(temp)
        cnt += 1
elif ratio < 1.0:
    for i, temp in enumerate(temps):
        if cnt > int(i * ratio):
            continue
            
        frames.append(temp)
        cnt += 1
            
        if cnt == frame_cnt:
            break
else:
    frames = temps[:frame_cnt]

print len(frames)
print frames
