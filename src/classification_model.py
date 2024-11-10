import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class LightTrafficViolationClassifier(nn.Module):
    def __init__(self, input_height, input_width, num_classes, hidden_size=64):
        super(LightTrafficViolationClassifier, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # Облегченные сверточные слои с меньшим количеством фильтров
        self.conv_layers = nn.Sequential(
            # Первый блок: уменьшаем количество фильтров до 16
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Второй блок: увеличиваем до 32 вместо 128
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Добавляем еще один слой с stride=2 вместо MaxPool
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # Вычисляем размер выхода после свёрточных слоев
        conv_output_height = input_height // 8
        conv_output_width = input_width // 8
        self.conv_output_size = 32 * conv_output_height * conv_output_width
        
        # Уменьшаем размер скрытого состояния
        self.frame_features = nn.Sequential(
            nn.Linear(self.conv_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)  # Уменьшаем dropout
        )
        
        # Упрощаем LSTM: один слой вместо двух
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,  # Уменьшаем количество слоев
            dropout=0,     # Убираем dropout в LSTM
            batch_first=True,
        )
        
        # Упрощаем классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.LogSoftmax(dim=-1)
        )
    
    def forward(self, x, lengths=None):
        batch_size, seq_length = x.size(0), x.size(1)
        
        # Оптимизируем обработку последовательности
        # Объединяем batch и sequence dimensions для CNN
        x = x.reshape(-1, 3, self.input_height, self.input_width)
        
        # CNN feature extraction
        x = self.conv_layers(x)
        
        # Reshape для полносвязного слоя
        x = x.reshape(batch_size * seq_length, -1)
        
        # Extract frame features
        frame_features = self.frame_features(x)
        
        # Reshape для LSTM
        frame_features = frame_features.reshape(batch_size, seq_length, -1)
        
        # Применяем LSTM
        if lengths is not None:
            # Если предоставлены длины последовательностей, используем pack_padded
            packed_sequence = nn.utils.rnn.pack_padded_sequence(
                frame_features, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed_sequence)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        else:
            lstm_out, _ = self.lstm(frame_features)
        
        # Классификация
        predictions = self.classifier(lstm_out)
        
        return predictions

def save_model(model, path):
    """
    Сохраняет модель и её конфигурацию
    """
    state = {
        'model_state_dict': model.state_dict(),
        'input_height': model.input_height,
        'input_width': model.input_width,
        'num_classes': model.num_classes,
        'hidden_size': model.hidden_size
    }
    torch.save(state, path)

def load_model(path, device):
    """
    Загружает модель с сохраненной конфигурацией
    """
    state = torch.load(path, map_location=device)
    
    # Создаем модель с сохраненными параметрами
    model = LightTrafficViolationClassifier(
        input_height=state['input_height'],
        input_width=state['input_width'],
        num_classes=state['num_classes'],
        hidden_size=state['hidden_size']
    )
    
    # Загружаем веса
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device)
    
    return model


class TemporalViolationLoss(nn.Module):
    def __init__(
        self,
        tolerance_seconds=5,
        class_weights=None,
        focal_gamma=2.0,
        temporal_window_weight=0.4
    ):
        super(TemporalViolationLoss, self).__init__()
        self.tolerance_seconds = tolerance_seconds
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.temporal_window_weight = temporal_window_weight

    def create_temporal_mask(self, targets, seq_length):
        """
        Создает маску для временного окна вокруг каждого нарушения
        """
        device = targets.device
        mask = torch.zeros((seq_length, seq_length), device=device)
        
        for t in range(seq_length):
            if targets[t] > 0:  # если есть нарушение
                # Создаем окно вокруг нарушения
                start_idx = max(0, t - self.tolerance_seconds)
                end_idx = min(seq_length, t + self.tolerance_seconds + 1)
                mask[t, start_idx:end_idx] = 1.0
                
        return mask

    def temporal_soft_target_loss(self, predictions, targets, seq_length):
        """
        Вычисляет лосс с учетом временного окна
        """
        device = predictions.device
        batch_size = predictions.size(0)
        num_classes = predictions.size(-1)
        loss = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            for c in range(1, num_classes):  # пропускаем класс 0 (нет нарушения)
                # Создаем бинарную маску для текущего класса
                class_targets = (targets[b] == c).float()
                if class_targets.sum() == 0:
                    continue

                # Получаем предсказания для текущего класса
                class_preds = predictions[b, :, c]
                
                # Создаем временную маску
                temporal_mask = self.create_temporal_mask(class_targets, seq_length)
                
                # Вычисляем взвешенную ошибку
                target_probs = F.normalize(temporal_mask, p=1, dim=1)
                pred_probs = F.softmax(class_preds.unsqueeze(0).expand_as(target_probs), dim=1)
                
                # KL-дивергенция между распределениями
                kl_div = F.kl_div(
                    pred_probs.log(),
                    target_probs,
                    reduction='batchmean'
                )
                
                loss += kl_div * (self.class_weights[c] if self.class_weights is not None else 1.0)

        return loss / batch_size

    def focal_loss(self, predictions, targets):
        """
        Focal loss с поддержкой весов классов
        """
        ce_loss = F.cross_entropy(
            predictions,
            targets,
            weight=self.class_weights,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.focal_gamma
        return (focal_term * ce_loss).mean()

    def forward(self, predictions, targets, lengths=None):
        """
        Args:
            predictions: [batch_size, seq_length, num_classes] или [batch_size, num_classes]
            targets: [batch_size, seq_length] или [batch_size]
            lengths: [batch_size] - длины последовательностей
        """
        device = predictions.device
        
        if len(predictions.shape) == 2:  # если предсказания для одного фрейма
            return self.focal_loss(predictions, targets)

        batch_size, seq_length, num_classes = predictions.shape
        total_loss = torch.tensor(0.0, device=device)
        
        # Базовый focal loss
        for t in range(seq_length):
            frame_preds = predictions[:, t, :]
            frame_targets = targets[:, t]
            
            # Применяем маску валидных фреймов
            if lengths is not None:
                valid_mask = t < lengths
                if not valid_mask.any():
                    continue
                frame_preds = frame_preds[valid_mask]
                frame_targets = frame_targets[valid_mask]
            
            total_loss += self.focal_loss(frame_preds, frame_targets)

        # Добавляем temporal loss
        temporal_loss = self.temporal_soft_target_loss(
            predictions,
            targets,
            seq_length
        )
        
        # Комбинируем потери
        final_loss = (total_loss / seq_length + 
                     self.temporal_window_weight * temporal_loss)

        return final_loss
    

    def set_class_weights(self, class_counts):
        """
        Установка весов классов на основе их распределения в датасете
        """
        total_samples = sum(class_counts)
        self.alpha = torch.tensor([
            total_samples / (len(class_counts) * count) 
            for count in class_counts
        ])
        self.alpha = self.alpha / self.alpha.sum()  # нормализация


def train_model(model, train_loader, criterion, optimizer, device, 
                num_epochs=10, validation_loader=None, window_size=5,
                save_path=None, scheduler=None):
    """
    Обучает модель с сохранением лучшей версии по F1-score
    
    Args:
        model: модель для обучения
        train_loader: загрузчик обучающих данных
        criterion: функция потерь (CustomViolationLoss)
        optimizer: оптимизатор
        device: устройство для вычислений
        num_epochs: количество эпох
        validation_loader: загрузчик валидационных данных
        window_size: размер окна для оценки нарушений
        save_path: путь для сохранения лучшей модели
        scheduler: планировщик скорости обучения
    """
    model.train()
    best_f1 = 0.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    training_history = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        # Обучение
        for frames, labels in train_loader:
            frames, labels = frames.to(device), labels.to(device)
            batch_size, seq_len = frames.shape[:2]
            
            # Создаем тензор длин последовательностей
            # Предполагаем, что все кадры валидны (1 кадр = 1 секунда)
            sequence_lengths = torch.full((batch_size,), seq_len, 
                                       device=device, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(frames)
            
            # Передаем выходы в оригинальном формате [batch_size, seq_len, num_classes]
            loss = criterion(outputs, labels, sequence_lengths)
            
            # Проверяем на NaN
            if torch.isnan(loss):
                print(f"NaN loss detected in epoch {epoch+1}, batch {num_batches+1}")
                print(f"Sequence lengths: {sequence_lengths}")
                print(f"Output shape: {outputs.shape}")
                print(f"Labels shape: {labels.shape}")
                continue
                
            loss.backward()
            
            # Опционально: градиентный клиппинг
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
        
        epoch_loss = running_loss / num_batches if num_batches > 0 else float('inf')
        
        # Валидация
        if validation_loader:
            model.eval()
            precision, recall, f1 = evaluate_model(model, validation_loader, device, window_size)
            
            # Сохраняем метрики
            metrics = {
                'epoch': epoch + 1,
                'train_loss': epoch_loss,
                'val_precision': precision,
                'val_recall': recall,
                'val_f1': f1
            }
            training_history.append(metrics)
            
            print(f'Epoch {epoch+1}:')
            print(f'Training Loss: {epoch_loss:.4f}')
            print(f'Validation - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
            
            # Сохраняем лучшую модель
            if f1 > best_f1 and save_path:
                best_f1 = f1
                save_model(model, save_path)
                print(f'Saved best model with F1: {f1:.4f}')
            
            # Обновляем learning rate
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(f1)
                else:
                    scheduler.step()
                
                # Выводим текущий learning rate
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Current learning rate: {current_lr:.2e}')
        else:
            print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()
    
    # Возвращаем историю обучения
    return training_history

def evaluate_model(model, test_loader, device, window_size=5):
    """
    Оценивает модель с учетом временного окна
    """
    model.eval()
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_sequences = 0
    
    with torch.no_grad():
        for frames, labels in test_loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            
            # Получаем предсказания
            predicted = torch.argmax(outputs, dim=2)  # Используем argmax вместо max
            
            # Обрабатываем каждую последовательность в батче
            for i in range(len(labels)):
                true_seq = labels[i].cpu().numpy()
                pred_seq = predicted[i].cpu().numpy()
                
                # Находим нарушения в истинных и предсказанных последовательностях
                true_violations = find_violations_in_window(true_seq, window_size)
                pred_violations = find_violations_in_window(pred_seq, window_size)
                
                # Вычисляем метрики для текущей последовательности
                precision, recall, f1 = calculate_window_accuracy(
                    true_violations, pred_violations, window_size
                )
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                num_sequences += 1
                
                # Выводим информацию о текущей последовательности для отладки
                print(f"\nSequence {num_sequences}:")
                print(f"True violations: {true_violations}")
                print(f"Predicted violations: {pred_violations}")
                print(f"Metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    # Вычисляем средние метрики
    avg_precision = total_precision / num_sequences if num_sequences > 0 else 0
    avg_recall = total_recall / num_sequences if num_sequences > 0 else 0
    avg_f1 = total_f1 / num_sequences if num_sequences > 0 else 0
    
    return avg_precision, avg_recall, avg_f1


def find_violations_in_window(predictions, window_size=5):
    """
    Находит нарушения в окне заданного размера, исключая повторения
    """
    violations = []
    used_times = set()
    
    for t in range(len(predictions)):
        pred_class = predictions[t]
        if pred_class == 0:  # пропускаем класс "нет нарушения"
            continue
            
        # Проверяем, не находится ли текущее время в уже использованном окне
        skip = False
        for used_start, used_end in used_times:
            if t >= used_start and t <= used_end:
                skip = True
                break
        if skip:
            continue
            
        violations.append((t, int(pred_class)))
        # Добавляем временное окно в использованные
        used_times.add((max(0, t - window_size // 2), 
                       min(len(predictions) - 1, t + window_size // 2)))
        
    return violations

def calculate_window_accuracy(true_violations, pred_violations, window_size=5):
    """
    Рассчитывает точность с учетом временного окна
    """
    if not true_violations and not pred_violations:
        return 1.0, 1.0, 1.0  # Если нет нарушений и предсказаний, считаем это идеальным случаем
    
    if not true_violations:
        return 0.0, 1.0, 0.0  # Если нет истинных нарушений, но есть предсказания
    
    if not pred_violations:
        return 1.0, 0.0, 0.0  # Если нет предсказаний, но есть истинные нарушения
    
    tp = 0  # true positives
    used_true = set()
    used_pred = set()
    
    # Для каждого истинного нарушения ищем соответствующее предсказание
    for t_true, class_true in true_violations:
        best_distance = float('inf')
        best_match = None
        
        for t_pred, class_pred in pred_violations:
            if t_pred in used_pred:
                continue
                
            distance = abs(t_true - t_pred)
            if distance <= window_size and class_true == class_pred:
                if distance < best_distance:
                    best_distance = distance
                    best_match = t_pred
        
        if best_match is not None:
            tp += 1
            used_true.add(t_true)
            used_pred.add(best_match)
    
    precision = tp / len(pred_violations) if pred_violations else 0
    recall = tp / len(true_violations) if true_violations else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def predict_video(model, frames, device, window_size=5):
    """
    Делает предсказания для видео
    """
    model.eval()
    with torch.no_grad():
        frames = frames.to(device)
        outputs = model(frames)
        predicted = torch.argmax(outputs, dim=2)  # Используем argmax вместо max
        
        # Получаем предсказания для последовательности
        pred_seq = predicted[0].cpu().numpy()  # берем первый элемент, так как batch_size=1
        
        # Находим нарушения с учетом временного окна
        violations = find_violations_in_window(pred_seq, window_size)
        
        # Выводим информацию о предсказаниях для отладки
        print(f"\nPredicted sequence:")
        print(f"Raw predictions: {pred_seq}")
        print(f"Found violations: {violations}")
        
    return violations

def predict_single_video(model, frames, device='cuda'):
    """
    Обрабатывает одиночное видео
    
    Args:
        model: модель
        frames: тензор формы [sequence_length, 3, height, width]
        device: устройство для вычислений
    """
    model.eval()
    
    # Добавляем размерность батча
    frames = frames.unsqueeze(0).to(device)  # [1, seq_len, 3, H, W]
    
    with torch.no_grad():
        
        # Предсказание
        outputs = model(frames)
        predicted = torch.argmax(outputs, dim=2)  # [1, seq_len]
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Убираем размерность батча
        predictions = predicted[0].cpu().numpy()  # [seq_len]
        
        # Находим нарушения
        violations = find_violations_in_window(predictions)
        
    return violations