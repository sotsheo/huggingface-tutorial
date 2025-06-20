from collections import deque
import copy

# Khởi tạo khối Rubik đã giải (mỗi mặt là 9 ô cùng màu)
def create_solved_cube():
    return {
        'U': ['W'] * 9,
        'D': ['Y'] * 9,
        'F': ['G'] * 9,
        'B': ['B'] * 9,
        'L': ['O'] * 9,
        'R': ['R'] * 9,
    }

# Xoay mặt F theo chiều kim đồng hồ
def move_F(cube):
    f = cube['F']
    # Xoay các ô trong mặt F
    cube['F'] = [f[6], f[3], f[0],
                 f[7], f[4], f[1],
                 f[8], f[5], f[2]]

    # Lưu lại cạnh để không bị ghi đè
    u_copy = cube['U'][6:9]
    r_copy = [cube['R'][0], cube['R'][3], cube['R'][6]]
    d_copy = cube['D'][0:3]
    l_copy = [cube['L'][2], cube['L'][5], cube['L'][8]]

    # Cập nhật các cạnh
    # L -> U
    cube['U'][6], cube['U'][7], cube['U'][8] = l_copy[::-1]

    # D -> L
    cube['L'][2], cube['L'][5], cube['L'][8] = d_copy

    # R -> D
    cube['D'][0], cube['D'][1], cube['D'][2] = r_copy[::-1]

    # U -> R
    cube['R'][0], cube['R'][3], cube['R'][6] = u_copy

# In khối Rubik (chỉ mặt trước)
def print_cube(cube):
    print("F mặt (Front):")
    f = cube['F']
    print(f[0:3])
    print(f[3:6])
    print(f[6:9])
    print()

# Kiểm tra Rubik đã giải chưa
def is_solved(cube):
    return all(face.count(face[0]) == 9 for face in cube.values())

# Danh sách bước xoay cho BFS (ở đây chỉ có F để đơn giản)
MOVES = ['F']

# Áp dụng bước
def apply_move(cube, move):
    new_cube = copy.deepcopy(cube)
    if move == 'F':
        move_F(new_cube)
    return new_cube

# BFS tìm lời giải
def bfs_solve(start_cube):
    visited = set()
    queue = deque([(start_cube, [])])

    while queue:
        cube, path = queue.popleft()
        key = str(cube)
        if key in visited:
            continue
        visited.add(key)

        if is_solved(cube):
            return path

        for move in MOVES:
            new_cube = apply_move(cube, move)
            queue.append((new_cube, path + [move]))
    
    return None  # Không tìm thấy lời giải

# MAIN – Thực thi chương trình
if __name__ == "__main__":
    print("Khởi tạo Rubik...")
    cube = create_solved_cube()
    print("Đã giải? ", is_solved(cube))  # True

    print("Xoay F (1 bước)...")
    move_F(cube)
    print("Đã giải? ", is_solved(cube))  # False
    print_cube(cube)

    print("Đang giải bằng BFS...")
    solution = bfs_solve(cube)
    print("Giải pháp tìm được:", solution)
