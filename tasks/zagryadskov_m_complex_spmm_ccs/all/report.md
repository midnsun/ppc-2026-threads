# Отчёт по задаче

## "Умножение разреженных матриц. Элементы комплексного типа. Формат хранения матрицы - столбцовый (CCS)"

## Вариант №7. Технология ALL

### Национальный исследовательский Нижегородский государственный университет им. Н.И. Лобачевского

### Институт информационных технологий, математики и механики

### Направление подготовки: прикладная математика и информатика

**Выполнил:**
Студент группы 3823Б1ПМоп3
**Загрядсков М.А**

**Преподаватель:**
доцент кафедры ВВСП **Сысоев А.В.**

---

## Введение

В данной работе реализован и исследован параллельный алгоритм умножения двух разреженных матриц в формате
**CCS** (Compressed Column Storage, см. [1]) с элементами комплексного типа `std::complex<double>` с
использованием технологий MPI [2] и OpenMP [3]. Реализация этого умножения направлена на распределение данных и
вычислений по процессам и распределение вычислений на процессах по потокам для уменьшения времени работы
алгоритма.

---

## Постановка задачи

Необходимо для двух матриц в формате **CCS** вычислить их произведение и затем сохранить его в формате **CCS**.
Произведение двух матриц A = a[i, j], B = b[j, k] (i = 1, m; j = 1, n; k = 1, p; a[i, j] ∈ **C**; b[i, j] ∈ **C**)
равно матрице C со элементами, выражающимися по формуле: C = c[i, k] = Σ(a[i, j] * b[j, k]), j ∈ 1, n.
Матрицы считаются разреженными, если количество нулевых элементов в ней гораздо больше количества
ненулевых элементов, или (что эквивалентно) если количество ненулевых элементов в ней порядка
одной из размерностей.
В формате **CCS** разреженная матрица m x n с nz - количеством ненулевых элементов представляется в виде 3х массивов:

- Массив `row_ind` размера nz, содержащий координату строки соответствующего элемента;
- Массив `values` размера nz, содержащий значение соотвествующего ненулевого элемента;
- Массив `col_ptr` размера n + 1, содержащий индекс начала нового столбца в массивах `row_ind` и `values`;

В ходе выполнения задачи требуется:

1. Реализовать параллельную версию данного алгоритма с использованием MPI и OpenMP;
2. Проверить корректность реализации и провести экспериментальные замеры времени выполнения;
3. Сравнить время работы параллельной реализации с использованием различного количества потоков с
последовательной реализацией

---

## Описание алгоритма

Умножение выполняется по алгоритму, состоящему из нескольких этапов:

**С точки зрения процессов выполняется:**
- Пересылка матрицы **A** всем MPI-процессам с помощью функции `BcastCCS`;
- Распределение столбцов матрицы **B** на корневом процессе, создание локальных для процессов подматриц
**B**, называмых **local_B** и их рассылка процессам с помощью функции `ScatterB`;
- Параллельное умножение **A*local_B = local_C** на каждом процессе. При этом сама операция выполняется
параллельно на **общей памяти** с использованием потоков по **двухфазному алгоритму**, полностью
аналогичному алгоритму, использующемуся задаче для технологии OMP. Алгоритм описан ниже;
- Пересылка **local_C** корневому процессу и сборка этих подматриц в итоговую матрицу **C** с помощью
функции `GatherC`;
- После умножения в функции PostProcessing пайплайна происходит рассылка матрицы **C**
остальным процессам для обеспечения успешного прохождения тестов всеми процессами;

В данном алгоритме параллельно на каждом из процессов выполняется непосредственное умножение
**A*local_B = local_C**. Однако, копирование данных в подматрицы **local_B** и их рассылка,
а также получение подматриц **local_C** и вычисление префиксной суммы `col_ptr` для матрицы **C**
происходит последовательно только на корневом процессе, что может помешать линейному ускорениию
относительно последовательной версии при увеличении количества процессов.

**С точки зрения потоков:**
Выполняется умножение по алгоритму, состоящему из **двух фаз**. При этом обработка столбцов матрицы
**local_B** и вычисление результирующих столбцов **local_C** выполняется параллельно и независимо
в каждой из фаз.

**1. Символьная фаза (symbolic phase):**
Для каждого столбца правой матрицы определяется структура соответствующего столбца результата.
- Выполняется обход ненулевых элементов и формируется множество индексов строк. Для этого используются
локальные для потоков **вектора-индикаторы (marker)**;
- Вычисляется массив **col_ptr** результирующей матрицы;

**2. Численная фаза (numeric phase):**
Для каждого столбца правой матрицы вычисляется столбец итоговой матрицы.
- Для каждого столбца используются локальные для потоков вектора **acc** аккумулирования результата и
**marker** для исключения повторной обработки столбцов;
- Численный результат записывается в заранее выделенные векторы результирующей матрицы;

Параллелизм основан на использовании технологии **OpenMP** с использованием. При этом каждый поток
выполняет функции symbolic и numeric, работая со своим диапазоном столбцов от `jstart` до `jend`.

Использование двухфазного подхода позволяет избежать гонок данных или блокировок потоков при записи
численного результата в итоговую матрицу, поскольку после выделения памяти на символьной фазе каждый поток
будет заполнять свои столбцы результирующей матрицы независимо от других столбцов. 
Также при таком подходе максимальное количество операций производится параллельно в каждом потоке,
последовательно выполняется только выделение памяти для матрицы **C** и подсчет префиксной суммы
столбцов `col_ptr`.

---

## Результаты экспериментов и подтверждение корректности

Для **функционального тестирования** были выбраны несколько предпосчитанных примеров, проверяющих
краевые случаи работы алгоритма. Сравнение результатов выполнялось с учетом погрешности операций с
плавающей точкой, которая не должна превышать `1e-14`.

Для **оценки производительности** использовался генератор разреженных ленточных матриц 40000 на 40000,
ширина ленты 100, ненулевые значения в диапазоне от 1.0 до 2.0 для действительной и мнимой компонент.
Результаты тестирования представлены ниже:

| Версия алгоритма | 1+2 | 2+1 | 2+3 | 2+6 |
| ---------------: | -------: | -------: | --------: | ---------: |
| Последовательная | 0.785 | 0.785 | 0.785 | 0.785 |
| Параллельная (MPI + OpenMP) | 0.523 | 0.656 | 0.243 | 0.209 |
| Ускорение (раз) | 1.5 | 1.2 | 3.2 | 3.8 |

В приведённо таблице во втором, третьем и последующих столбцах представлено время выполнения в секундах
при использовании 2, 3, 6, и 12 потоков. Тестирование производилось на 6-ядерном процессоре Intel i5-12400f.

---

## Выводы из результатов

Реализация с использованием MPI и OpenMP показывает ускорение до **3.8 раз**. Это демонстрирует, что алгоритм
хорошо приспособлен к параллелизму, а его реализация является достаточно эффективной. Ожидаемо, параллелизм проявляет
себя лучше при увеличении количества процессов, нежели количества потоков.

---

## Заключение

Реализована и протестирована параллельная реализация алгоритма умножения двух разреженных матриц
в формате **CCS** с элементами комплексного типа с использованием технологий MPI и OpenMP.
В результате экспериментов подтверждена корректность реализации, вычислено время работы алгоритма
и проведено сравнение с последовательной реализацией.

---

## Список литературы

1. Лекция кафедры ВВСП о хранении разреженных матриц в формате презентации:
<https://hpc-education.unn.ru/files/courses/optimization/2_3_SparseDS_Lect.pdf>
2. Документация в формате официального веб-сайта к технологии MPI:
<https://www.mpich.org>
3. Документация к технологии OpenMP:
<https://www.openmp.org/wp-content/uploads/OpenMP-RefGuide-6.0-OMP60SC24-web.pdf>

---

## Приложение

Реализация параллельного алгоритма умножения двух комплексных матриц в формате CCS с
использованием технологий MPI и OpenMP:

```cpp
void ZagryadskovMComplexSpMMCCSALL::BcastCCS(CCS &a, int rank) {
  MPI_Bcast(&a.m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&a.n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int nz = 0;
  if (rank == 0) {
    nz = static_cast<int>(a.values.size());
  }
  MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    a.col_ptr.resize(a.n + 1);
    a.row_ind.resize(nz);
    a.values.resize(nz);
  }

  MPI_Bcast(a.col_ptr.data(), a.n + 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(a.row_ind.data(), nz, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(a.values.data(), nz, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
}

void ZagryadskovMComplexSpMMCCSALL::SendCCS(const CCS &m, int dest) {
  MPI_Send(&m.m, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
  MPI_Send(&m.n, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
  int nz = static_cast<int>(m.values.size());
  MPI_Send(&nz, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);

  MPI_Send(m.col_ptr.data(), m.n + 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
  MPI_Send(m.row_ind.data(), nz, MPI_INT, dest, 0, MPI_COMM_WORLD);
  MPI_Send(m.values.data(), nz, MPI_C_DOUBLE_COMPLEX, dest, 0, MPI_COMM_WORLD);
}

void ZagryadskovMComplexSpMMCCSALL::RecvCCS(CCS &m, int src) {
  MPI_Status st;
  MPI_Recv(&m.m, 1, MPI_INT, src, 0, MPI_COMM_WORLD, &st);
  MPI_Recv(&m.n, 1, MPI_INT, src, 0, MPI_COMM_WORLD, &st);
  int nz = 0;
  MPI_Recv(&nz, 1, MPI_INT, src, 0, MPI_COMM_WORLD, &st);

  m.col_ptr.resize(m.n + 1);
  m.row_ind.resize(nz);
  m.values.resize(nz);

  MPI_Recv(m.col_ptr.data(), m.n + 1, MPI_INT, src, 0, MPI_COMM_WORLD, &st);
  MPI_Recv(m.row_ind.data(), nz, MPI_INT, src, 0, MPI_COMM_WORLD, &st);
  MPI_Recv(m.values.data(), nz, MPI_C_DOUBLE_COMPLEX, src, 0, MPI_COMM_WORLD, &st);
}

void ZagryadskovMComplexSpMMCCSALL::ScatterB(const CCS &b, CCS &b_local, const std::vector<int> &col_starts, int rank,
                                             int size) {
  if (rank == 0) {
    CCS tmp;
    for (int proc = 0; proc < size; ++proc) {
      int jstart = col_starts[proc];
      int jend = col_starts[proc + 1];

      tmp.m = b.m;
      tmp.n = jend - jstart;
      tmp.row_ind.clear();
      tmp.values.clear();
      tmp.col_ptr.clear();

      int nnz_start = b.col_ptr[jstart];
      int nnz_end = b.col_ptr[jend];
      tmp.row_ind.assign(b.row_ind.begin() + nnz_start, b.row_ind.begin() + nnz_end);
      tmp.values.assign(b.values.begin() + nnz_start, b.values.begin() + nnz_end);
      tmp.col_ptr.resize(tmp.n + 1);
      for (int j = 0; j <= tmp.n; ++j) {
        tmp.col_ptr[j] = b.col_ptr[jstart + j] - nnz_start;
      }

      if (proc == 0) {
        b_local = tmp;
      } else {
        SendCCS(tmp, proc);
      }
    }
  } else {
    RecvCCS(b_local, 0);
  }
}

void ZagryadskovMComplexSpMMCCSALL::GatherC(CCS &c, CCS &c_local, int rank, int size) {
  MPI_Status st;
  int local_nnz = static_cast<int>(c_local.values.size());
  int total_nnz = 0;
  int local_cols = c_local.n;
  int total_cols = 0;
  std::vector<int> tmp;
  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);

  MPI_Gather(&local_nnz, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    c.m = c_local.m;
    for (int i = 0; i < size; ++i) {
      displs[i] = total_nnz;
      total_nnz += recvcounts[i];
    }
    c.row_ind.resize(total_nnz);
    c.values.resize(total_nnz);
  }

  MPI_Gatherv(c_local.row_ind.data(), local_nnz, MPI_INT, c.row_ind.data(), recvcounts.data(), displs.data(), MPI_INT,
              0, MPI_COMM_WORLD);
  MPI_Gatherv(c_local.values.data(), local_nnz, MPI_C_DOUBLE_COMPLEX, c.values.data(), recvcounts.data(),
              displs.data(), MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

  MPI_Gather(&local_cols, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      displs[i] = total_cols + 1;
      total_cols += recvcounts[i];
      recvcounts[i] += 1;
    }
    c.n = total_cols;
    c.col_ptr.resize(total_cols + 1);
  }

  if (rank != 0) {
    MPI_Send(c_local.col_ptr.data(), c_local.n + 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    std::ranges::copy(c_local.col_ptr, c.col_ptr.begin());

    int nz_offset = c_local.col_ptr.back();
    int col_offset = c_local.n;
    for (int proc = 1; proc < size; ++proc) {
      tmp.resize(recvcounts[proc]);
      MPI_Recv(tmp.data(), recvcounts[proc], MPI_INT, proc, 0, MPI_COMM_WORLD, &st);

      for (int j = 1; j < recvcounts[proc]; ++j) {
        c.col_ptr[col_offset + j] = nz_offset + tmp[j];
      }

      nz_offset += tmp.back();
      col_offset += recvcounts[proc] - 1;
      tmp.clear();
    }
  }
}

bool ZagryadskovMComplexSpMMCCSALL::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  CCS &a = std::get<0>(GetInput());
  CCS &b = std::get<1>(GetInput());
  CCS &c = GetOutput();

  CCS local_b;
  CCS local_c;
  std::vector<int> col_starts;
  if (world_rank == 0) {
    col_starts.resize(world_size + 1);
    for (int proc = 0; proc <= world_size; ++proc) {
      col_starts[proc] = (proc * b.n) / world_size;
    }
  }

  BcastCCS(a, world_rank);
  ScatterB(b, local_b, col_starts, world_rank, world_size);

  ZagryadskovMComplexSpMMCCSALL::SpMM(a, local_b, local_c);

  GatherC(c, local_c, world_rank, world_size);

  return true;
}
```
